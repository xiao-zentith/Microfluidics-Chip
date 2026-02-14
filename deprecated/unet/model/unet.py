import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    基础卷积块: (Conv3x3 -> BN -> GELU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 尺寸校验 (防止奇数尺寸导致的拼接错误)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # x1: Decoder feature, x2: Encoder Skip Connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class EnvironmentEncoder(nn.Module):
    """
    分支2：环境参考编码器 (Reference Stream)
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.inc = DoubleConv(in_channels, features[0])
        self.downs = nn.ModuleList()
        in_ch = features[0]
        for feat in features[1:]:
            self.downs.append(Down(in_ch, feat))
            in_ch = feat
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.inc(x)
        for down in self.downs:
            x = down(x)
        return self.global_pool(x)

class RefGuidedUNet(nn.Module):
    """
    论文核心模型: Dual-Stream U-Net
    """
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        
        # --- Stream 1: Signal Encoder ---
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        # --- Stream 2: Reference Encoder ---
        self.ref_encoder = EnvironmentEncoder(in_channels, features)
        
        # --- Fusion ---
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(features[3] * 2, features[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(features[3]),
            nn.GELU()
        )
        
        # --- Decoder ---
        self.up1 = Up(features[3] + features[2], features[2])
        self.up2 = Up(features[2] + features[1], features[1])
        self.up3 = Up(features[1] + features[0], features[0])
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x_signal, x_ref):
        # 1. 信号流
        x1 = self.inc(x_signal)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # 2. 参考流
        env_vec = self.ref_encoder(x_ref)
        
        # 3. 融合
        env_map = env_vec.expand_as(x4)
        fused = torch.cat([x4, env_map], dim=1)
        bottleneck = self.fusion_conv(fused)
        
        # 4. 解码
        x = self.up1(bottleneck, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        return torch.sigmoid(self.outc(x))

class ROIWeightedLoss(nn.Module):
    def __init__(self, roi_radius=20, edge_weight=0.1, lambda_cos=0.2):
        """
        Args:
            roi_radius: 核心反应区半径 (像素)，该区域权重为 1.0。
                        注意：如果切片变大了，只要腔室没变大，这个值通常不用变。
            edge_weight: 边缘/背景区域的权重 (0.1)，容忍对齐误差
            lambda_cos: 余弦相似度损失的权重
        """
        super().__init__()
        self.roi_radius = roi_radius
        self.edge_weight = edge_weight
        self.lambda_cos = lambda_cos
        
        # 移除 __init__ 中的固定 weight_map 生成
        # 改为缓存机制
        self.weight_map = None
        self.current_size = None

    def _create_weight_map(self, size, r, w_edge, device):
        """动态生成权重图"""
        center = size // 2
        Y, X = torch.meshgrid(torch.arange(size, device=device), 
                              torch.arange(size, device=device), 
                              indexing='ij')
        dist = torch.sqrt((X - center)**2 + (Y - center)**2)
        
        # ROI区域权重=1.0，边缘权重=w_edge
        mask = torch.where(dist <= r, torch.tensor(1.0, device=device), torch.tensor(w_edge, device=device))
        
        # 扩展维度 [H, W] -> [1, 1, H, W] 以进行广播
        return mask.view(1, 1, size, size)

    def forward(self, pred, target):
        # 获取当前输入的尺寸 (假设是方形 H=W)
        batch, channel, h, w = pred.shape
        
        # 如果尺寸变了，或者还没生成过 map，就重新生成
        if self.weight_map is None or self.current_size != h or self.weight_map.device != pred.device:
            self.current_size = h
            self.weight_map = self._create_weight_map(h, self.roi_radius, self.edge_weight, pred.device)
        
        # 2. ROI Weighted MSE Loss (光度/强度准确性)
        # 使用 MSE 而不是 L1，因为 MSE 对大误差(光照梯度)惩罚更重
        loss_pixel = torch.mean(self.weight_map * (pred - target) ** 2)
        
        # 3. Cosine Similarity Loss (光谱/浓度准确性)
        # 保证 RGB 向量的方向一致，即 R/G 比值正确
        cos_sim = F.cosine_similarity(pred, target, dim=1, eps=1e-8)
        loss_cos = 1.0 - cos_sim.mean()
        
        # 总损失
        total_loss = loss_pixel + self.lambda_cos * loss_cos
        
        return total_loss, loss_pixel, loss_cos