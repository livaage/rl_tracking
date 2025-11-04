"""
Logging utility for track propagation debugging.
Writes debug output to a log file instead of stdout.
"""
import logging
from pathlib import Path
from datetime import datetime

# Global logger instance
_logger = None

def get_logger(log_file=None, level=logging.DEBUG):
    """
    Get or create the global logger instance.
    
    Args:
        log_file: Path to log file. If None, uses 'propagation_debug.log' in current directory
        level: Logging level (default: DEBUG)
    
    Returns:
        Logger instance
    """
    global _logger
    
    if _logger is None:
        if log_file is None:
            log_file = Path.cwd() / 'propagation_debug.log'
        else:
            log_file = Path(log_file)
        
        # Create logger
        _logger = logging.getLogger('propagation')
        _logger.setLevel(level)
        _logger.handlers = []  # Clear existing handlers
        
        # File handler - logs all levels (DEBUG, INFO, WARNING, ERROR)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        _logger.addHandler(file_handler)
        
        # Console handler - only logs ERROR and CRITICAL to avoid cluttering console with propagation messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(formatter)
        _logger.addHandler(console_handler)
    
    return _logger

def set_log_file(log_file):
    """Set a new log file for the logger."""
    global _logger
    if _logger is not None:
        # Remove existing file handlers
        _logger.handlers = [h for h in _logger.handlers if not isinstance(h, logging.FileHandler)]
        
        # Add new file handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)
    else:
        get_logger(log_file)

def debug(msg):
    """Log a debug message."""
    logger = get_logger()
    logger.debug(msg)

def info(msg):
    """Log an info message."""
    logger = get_logger()
    logger.info(msg)

def warning(msg):
    """Log a warning message."""
    logger = get_logger()
    logger.warning(msg)

def error(msg):
    """Log an error message."""
    logger = get_logger()
    logger.error(msg)

