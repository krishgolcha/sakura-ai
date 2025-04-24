import time
from typing import Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CanvasRateLimiter:
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests_per_minute = max_requests_per_minute
        self.request_times = []
        self.last_reset = datetime.now()
        self.priority_queues = {
            'high': [],    # Critical requests (e.g., assignments, grades)
            'medium': [],  # Important requests (e.g., announcements)
            'low': []      # Less critical requests (e.g., syllabus, modules)
        }
        
    def wait_if_needed(self, priority: str = 'medium') -> None:
        """Wait if we're approaching rate limit, with priority-based queuing"""
        now = datetime.now()
        
        # Reset request times if a minute has passed
        if now - self.last_reset > timedelta(minutes=1):
            self.request_times = []
            self.last_reset = now
            self.priority_queues = {k: [] for k in self.priority_queues}
            
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < timedelta(minutes=1)]
        
        # Calculate available slots
        available_slots = self.max_requests_per_minute - len(self.request_times)
        
        # Handle priority-based queuing
        if priority == 'high':
            # High priority requests get immediate access if slots available
            if available_slots > 0:
                return
        elif priority == 'medium':
            # Medium priority requests wait if high priority queue is not empty
            if self.priority_queues['high'] and available_slots <= 2:
                wait_time = 60 - (now - self.request_times[0]).total_seconds()
                if wait_time > 0:
                    logger.info(f"Medium priority request waiting {wait_time:.2f} seconds")
                    time.sleep(wait_time)
                    self.request_times = []
                    self.last_reset = datetime.now()
                    return
        else:  # low priority
            # Low priority requests wait if higher priority queues are not empty
            if (self.priority_queues['high'] or self.priority_queues['medium']) and available_slots <= 1:
                wait_time = 60 - (now - self.request_times[0]).total_seconds()
                if wait_time > 0:
                    logger.info(f"Low priority request waiting {wait_time:.2f} seconds")
                    time.sleep(wait_time)
                    self.request_times = []
                    self.last_reset = datetime.now()
                    return
        
        # If we're close to the limit, wait
        if len(self.request_times) >= self.max_requests_per_minute * 0.9:  # 90% of limit
            wait_time = 60 - (now - self.request_times[0]).total_seconds()
            if wait_time > 0:
                logger.info(f"Approaching rate limit, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                self.request_times = []
                self.last_reset = datetime.now()
                
    def add_request(self, priority: str = 'medium') -> None:
        """Record a new request with priority"""
        self.request_times.append(datetime.now())
        self.priority_queues[priority].append(datetime.now())
        
    def handle_rate_limit(self, response) -> Optional[float]:
        """Handle rate limit response from Canvas API"""
        if response.status_code == 429:  # Too Many Requests
            retry_after = float(response.headers.get('Retry-After', 5))
            logger.warning(f"Rate limit hit, waiting {retry_after} seconds")
            time.sleep(retry_after)
            return retry_after
        return None 