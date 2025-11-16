"""Logging configuration for AAG pattern matching"""

import logging
import sys
from typing import Optional

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        level: Logging level (default INFO)
        log_file: Optional log file path
        format_string: Optional custom format string
    
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True
    )
    
    return logging.getLogger(__name__)

def get_logger(name: str) -> logging.Logger:
    """Get logger for module"""
    return logging.getLogger(name)
