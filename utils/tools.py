# utils.py

import time
from contextlib import contextmanager

@contextmanager
def timer(log_values, tau=0.99, n=100):
    """Context manager to time a code block and update log values with min, max, and moving average."""

    # Only update every nth iteration
    log_values['counter'] += 1
    if log_values['counter'] < n:
        yield  # No timing needed for this iteration
        return

    # Start timing
    start_time = time.perf_counter()
    yield  # Run the block of code inside the `with` statement
    end_time = time.perf_counter()

    duration = end_time - start_time

    # Handle the first-time setting of values
    if log_values['min'] is None:
        log_values['min'] = duration
        log_values['max'] = duration
        log_values['moving_avg'] = duration
    else:
        # Update minimum and maximum
        log_values['min'] = min(log_values['min'], duration)
        log_values['max'] = max(log_values['max'], duration)

        # Update moving average with exponential weighting
        log_values['moving_avg'] = tau * log_values['moving_avg'] + (1 - tau) * duration

    log_values['counter'] = 0  # Reset the counter after the update
