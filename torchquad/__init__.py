import os
"""Set main device by default to cpu if no other choice was made before"""
if "TORCH_DEVICE" not in os.environ:
        os.environ["TORCH_DEVICE"] = 'cpu'