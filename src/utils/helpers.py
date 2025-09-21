from typing import Callable, TypeVar, List, Any, Optional, Iterator, Dict
from functools import wraps, reduce, partial
import logging
from pathlib import Path

T = TypeVar('T')
U = TypeVar('U')

def compose(*functions: Callable) -> Callable:
    """Compose multiple functions into a single function (right to left)."""
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def pipe(data: T, *functions: Callable[[T], T]) -> T:
    """Pipe data through a series of functions (left to right)."""
    return reduce(lambda acc, func: func(acc), functions, data)

def curry(func: Callable) -> Callable:
    """Convert a function to curried form."""
    @wraps(func)
    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)
        return lambda *more_args, **more_kwargs: curried(*(args + more_args), **{**kwargs, **more_kwargs})
    return curried

def memoize(func: Callable) -> Callable:
    """Memoize function results for performance optimization."""
    cache = {}
    @wraps(func)
    def memoized(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return memoized

def safe_execute(func: Callable, default: Any = None) -> Callable:
    """Create a safe version of a function that returns default on exception."""
    @wraps(func)
    def safe_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning(f"Function {func.__name__} failed: {e}")
            return default
    return safe_func

def chunk_list(lst: List[T], chunk_size: int) -> Iterator[List[T]]:
    """Split a list into chunks of specified size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def flatten(nested_list: List[List[T]]) -> List[T]:
    """Flatten a nested list structure."""
    return [item for sublist in nested_list for item in sublist]

def filter_none(lst: List[Optional[T]]) -> List[T]:
    """Remove None values from a list."""
    return [item for item in lst if item is not None]

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def batch_process(items: List[T], process_func: Callable[[T], U], batch_size: int = 32) -> List[U]:
    """Process items in batches for memory efficiency."""
    results = []
    for batch in chunk_list(items, batch_size):
        batch_results = [process_func(item) for item in batch]
        results.extend(batch_results)
    return results

def validate_path(path: str, create_if_missing: bool = False) -> Path:
    """Validate and optionally create a path."""
    path_obj = Path(path)
    if create_if_missing and not path_obj.exists():
        path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj
