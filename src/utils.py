import functools
import time
import torch

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_t = time.perf_counter()
        f_value = func(*args, **kwargs)
        elapsed_t = time.perf_counter() - start_t
        mins = elapsed_t // 60
        print(
            f"'{func.__name__}' elapsed time: {mins} minutes, {elapsed_t - mins * 60:0.2f} seconds"
        )
        return f_value

    return wrapper_timer


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.has_mps:
        return torch.device("mps")
    else:
        return torch.device("cpu")