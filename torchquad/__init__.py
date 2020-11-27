import os

"""Set main device by default to cpu if no other choice was made before"""
if "TORCH_DEVICE" not in os.environ:
    os.environ["TORCH_DEVICE"] = "cpu"


# Currently this is the way to expose to the docs
# hopefully changes with setup.py
from .integration.base_integrator import BaseIntegrator

__all__ = ["BaseIntegrator"]

