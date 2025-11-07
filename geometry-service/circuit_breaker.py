"""
Circuit Breaker Pattern for AAGNet Feature Recognition
Prevents cascade failures by opening circuit after repeated failures
"""

import time
import logging
from enum import Enum
from typing import Callable, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Circuit broken, rejecting requests
    HALF_OPEN = "HALF_OPEN"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascade failures.
    
    States:
    - CLOSED: Normal operation, allows all requests
    - OPEN: After threshold failures, blocks all requests for timeout period
    - HALF_OPEN: After timeout, allows one test request to check recovery
    
    Args:
        failure_threshold: Number of consecutive failures before opening circuit
        timeout_seconds: How long to keep circuit open before testing recovery
        expected_exception: Exception type that counts as failure (default: Exception)
    """
    
    def __init__(
        self, 
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
        
        logger.info(
            f"Circuit breaker initialized: "
            f"threshold={failure_threshold}, timeout={timeout_seconds}s"
        )
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
        """
        
        # Check if we should test recovery
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info("Circuit breaker entering HALF_OPEN state (testing recovery)")
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker is OPEN. "
                    f"Last failure: {self.failure_count} consecutive failures. "
                    f"Retry after {self._get_retry_after()} seconds."
                )
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Success - reset failure count
            self._on_success()
            return result
            
        except self.expected_exception as e:
            # Failure - increment counter
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to test recovery"""
        if self.last_failure_time is None:
            return True
        
        elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout_seconds
    
    def _get_retry_after(self) -> int:
        """Calculate seconds until circuit can be tested"""
        if self.last_failure_time is None:
            return 0
        
        elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return max(0, int(self.timeout_seconds - elapsed))
    
    def _on_success(self):
        """Handle successful function execution"""
        if self.state == CircuitState.HALF_OPEN:
            logger.info("✅ Circuit breaker recovery successful - returning to CLOSED state")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
        elif self.failure_count > 0:
            logger.info(f"Circuit breaker: Success after {self.failure_count} failures, resetting counter")
            self.failure_count = 0
            self.last_failure_time = None
    
    def _on_failure(self):
        """Handle function execution failure"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.state == CircuitState.HALF_OPEN:
            logger.warning("⚠️ Circuit breaker: Recovery test FAILED - returning to OPEN state")
            self.state = CircuitState.OPEN
        elif self.failure_count >= self.failure_threshold:
            logger.error(
                f"❌ Circuit breaker OPENED after {self.failure_count} consecutive failures. "
                f"Blocking requests for {self.timeout_seconds} seconds."
            )
            self.state = CircuitState.OPEN
        else:
            logger.warning(
                f"⚠️ Circuit breaker: Failure {self.failure_count}/{self.failure_threshold}"
            )
    
    def reset(self):
        """Manually reset circuit breaker to CLOSED state"""
        logger.info("Circuit breaker manually reset to CLOSED state")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
    
    def get_state(self) -> dict:
        """Get current circuit breaker state for monitoring"""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'retry_after_seconds': self._get_retry_after() if self.state == CircuitState.OPEN else 0
        }


# Global circuit breaker instance for AAGNet
aagnet_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    timeout_seconds=60,
    expected_exception=Exception
)
