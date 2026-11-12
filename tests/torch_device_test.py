"""Which torchquad setup calls may change PyTorch's default device.

Only ``enable_cuda`` (and ``set_up_backend`` with ``torch_enable_cuda=True``,
which calls it) may move the default device, and only to CUDA. The checks that
need a GPU run when one is present; on a CPU-only machine the same tests assert
that the device stays on the CPU.
"""

import warnings

import pytest
import torch

from torchquad import Simpson, enable_cuda, set_precision, set_up_backend


@pytest.fixture
def restore_torch_defaults():
    """Start on the CPU default and put PyTorch's global defaults back afterwards.

    Other test modules leave CUDA as the default device, and both the device
    and the dtype are process-wide.
    """
    device_before = torch.empty(0).device
    dtype_before = torch.get_default_dtype()
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(device_before)
    torch.set_default_dtype(dtype_before)


def _initialize_cuda_if_present():
    """Touch the GPU the way unrelated user code would, without changing defaults."""
    if torch.cuda.is_available():
        torch.ones(3, device="cuda")
        assert torch.cuda.is_initialized()


def test_set_up_backend_without_cuda_leaves_the_device_alone(restore_torch_defaults):
    """torch_enable_cuda=False must hold even after the GPU has been used.

    Regression test: set_precision moved the default device to CUDA whenever
    CUDA was initialized, so the flag was ignored as soon as any earlier code
    in the process had touched the GPU.
    """
    _initialize_cuda_if_present()

    set_up_backend("torch", "float64", torch_enable_cuda=False)

    assert torch.empty(0).device.type == "cpu"
    result = Simpson().integrate(
        lambda points: points[:, 0], dim=1, N=11, integration_domain=[[0.0, 1.0]]
    )
    assert result.device.type == "cpu"
    assert result.dtype == torch.float64


def test_set_precision_never_changes_the_device(restore_torch_defaults):
    """set_precision sets the dtype only, whatever state CUDA is in."""
    _initialize_cuda_if_present()

    set_precision("float64")

    assert torch.empty(0).device.type == "cpu"
    assert torch.get_default_dtype() == torch.float64


@pytest.mark.parametrize("data_type", [None, "float64"])
def test_enable_cuda_selects_the_gpu_when_there_is_one(restore_torch_defaults, data_type):
    """enable_cuda makes CUDA the default device, with or without a dtype.

    With ``data_type=None`` it used to initialize CUDA but leave the default
    device on the CPU, because only set_precision switched the device.
    """
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    with warnings.catch_warnings():
        # On a CPU-only machine enable_cuda warns that it falls back to the CPU.
        warnings.simplefilter("ignore", RuntimeWarning)
        enable_cuda(data_type=data_type)

    assert torch.empty(0).device.type == expected


def test_set_up_backend_selects_the_gpu_by_default(restore_torch_defaults):
    """The default set_up_backend call still puts integrations on the GPU."""
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        set_up_backend("torch", "float64")

    result = Simpson().integrate(
        lambda points: points[:, 0], dim=1, N=11, integration_domain=[[0.0, 1.0]]
    )
    assert result.device.type == expected


@pytest.mark.parametrize("data_type", ["float16", "float128", "int32", ""])
def test_set_precision_rejects_unsupported_data_types(restore_torch_defaults, data_type):
    """An unsupported precision must raise instead of quietly becoming float32.

    The fallback was a silent precision downgrade for anyone asking for more
    than float32, announced only by a line on stderr.
    """
    torch.set_default_dtype(torch.float64)
    with pytest.raises(ValueError, match="Only float32 and float64 are supported"):
        set_precision(data_type)
    assert torch.get_default_dtype() == torch.float64


def test_set_precision_rejects_unknown_backends():
    """An unknown backend must raise rather than print and carry on."""
    with pytest.raises(ValueError, match="not supported for backend"):
        set_precision("float64", backend="cupy")


@pytest.mark.parametrize("alias,expected", [("double", torch.float64), ("Float", torch.float32)])
def test_set_precision_keeps_the_legacy_aliases(restore_torch_defaults, alias, expected):
    """ "float" and "double", in any case, stay accepted for backwards compatibility."""
    set_precision(alias)
    assert torch.get_default_dtype() == expected
