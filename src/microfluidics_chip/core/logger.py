"""
统一日志系统

提供：
- 彩色控制台输出
- 文件日志记录
- 结构化日志格式
- 多级别日志（DEBUG, INFO, WARNING, ERROR）
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console


def setup_logger(
    name: str = "microfluidics_chip",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    use_rich: bool = True
) -> logging.Logger:
    """
    设置日志记录器
    
    :param name: 日志记录器名称
    :param level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
    :param log_file: 日志文件路径（可选）
    :param use_rich: 是否使用 Rich 美化控制台输出
    :return: 配置好的 logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    # 格式化字符串
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 控制台 handler
    if use_rich:
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=False,  # Rich 会自动显示时间
            show_path=False
        )
        console_handler.setLevel(logging.INFO)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(log_format, datefmt=date_format)
        console_handler.setFormatter(console_formatter)
    
    logger.addHandler(console_handler)
    
    # 文件 handler（如果指定）
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
    
    return logger


# 默认全局 logger
default_logger = setup_logger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取 logger 实例
    
    :param name: 子模块名称（可选），如 "stage1_detection"
    :return: logger 实例
    """
    if name:
        return logging.getLogger(f"microfluidics_chip.{name}")
    return default_logger
