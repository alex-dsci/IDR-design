# Thank you Stack Overflow user Izaya
# Code grabbed from: https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish

import errno, os, signal, functools
from typing import Callable 

DEFAULT_TIMEOUT_MESSAGE = os.strerror(errno.ETIME)
def timeout(seconds: int, message: str = DEFAULT_TIMEOUT_MESSAGE):
    def decorator(func: Callable):
        def _handle_timeout(signum, frame):
            raise RuntimeError(message)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result    
        return wrapper
    return decorator
