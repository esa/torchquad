import warnings


def _torch_trace_without_warnings(*args, **kwargs):
    """Execute `torch.jit.trace` on the passed arguments and hide tracer warnings

    PyTorch can show warnings about traces being potentially incorrect because
    the Python3 control flow is not completely recorded.
    This function can be used to hide the warnings in situations where they are
    false positives.
    """
    import torch

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        return torch.jit.trace(*args, **kwargs)
