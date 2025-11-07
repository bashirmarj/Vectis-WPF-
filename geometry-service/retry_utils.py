"""
Retry utilities with exponential backoff for transient errors
"""

import time
import logging
from typing import Callable, Any, Type, Tuple
from functools import wraps

logger = logging.getLogger(__name__)


class TransientError(Exception):
    """Errors that warrant retry with exponential backoff"""
    pass


class PermanentError(Exception):
    """Errors that should not be retried"""
    pass


class SystemicError(Exception):
    """Errors requiring immediate operational team alert"""
    pass


def exponential_backoff_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
    transient_exceptions: Tuple[Type[Exception], ...] = (TransientError,)
):
    """
    Decorator for retry with exponential backoff.
    
    Retry delays: 1s, 2s, 4s, 8s (capped at max_delay)
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        transient_exceptions: Exception types that trigger retry
        
    Example:
        @exponential_backoff_retry(max_attempts=3, base_delay=1.0)
        def process_cad_file(file_path):
            # May raise TransientError
            pass
    """
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            attempt = 0
            last_exception = None
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                    
                except transient_exceptions as e:
                    attempt += 1
                    last_exception = e
                    
                    if attempt >= max_attempts:
                        logger.error(
                            f"❌ {func.__name__} failed after {max_attempts} attempts: {str(e)}"
                        )
                        raise
                    
                    # Calculate exponential backoff delay
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    
                    logger.warning(
                        f"⚠️ {func.__name__} attempt {attempt}/{max_attempts} failed: {str(e)}. "
                        f"Retrying in {delay}s..."
                    )
                    
                    time.sleep(delay)
                
                except (PermanentError, SystemicError) as e:
                    # Don't retry permanent or systemic errors
                    logger.error(f"❌ {func.__name__} permanent error (no retry): {str(e)}")
                    raise
                    
                except Exception as e:
                    # Unknown exceptions don't retry by default
                    logger.error(f"❌ {func.__name__} unknown error (no retry): {str(e)}")
                    raise
            
            # Should never reach here
            raise last_exception
        
        return wrapper
    return decorator


def classify_error(exception: Exception) -> Type[Exception]:
    """
    Classify exception as transient, permanent, or systemic.
    
    Args:
        exception: The exception to classify
        
    Returns:
        TransientError, PermanentError, or SystemicError type
    """
    
    error_message = str(exception).lower()
    
    # Transient errors (retry with backoff)
    transient_indicators = [
        'timeout',
        'connection',
        'network',
        'temporarily unavailable',
        'resource exhaustion',
        'too many requests'
    ]
    
    # Permanent errors (no retry)
    permanent_indicators = [
        'invalid',
        'malformed',
        'validation failed',
        'parse error',
        'corrupt',
        'unsupported format'
    ]
    
    # Systemic errors (alert operations)
    systemic_indicators = [
        'model load failed',
        'out of memory',
        'cuda error',
        'disk full',
        'permission denied'
    ]
    
    if any(indicator in error_message for indicator in systemic_indicators):
        return SystemicError
    elif any(indicator in error_message for indicator in permanent_indicators):
        return PermanentError
    elif any(indicator in error_message for indicator in transient_indicators):
        return TransientError
    else:
        # Default to permanent (don't retry unknown errors)
        return PermanentError
