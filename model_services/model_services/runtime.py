import os
from enum import StrEnum

import torch


class ModelDevice(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


def positive_env_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def resolve_device(name: str) -> str:
    try:
        configured = ModelDevice(os.getenv(name, ModelDevice.AUTO))
    except ValueError as exc:
        raise ValueError(f"{name} must be one of {', '.join(ModelDevice)}") from exc
    if configured is not ModelDevice.AUTO:
        return configured.value
    if torch.cuda.is_available():
        return ModelDevice.CUDA.value
    if torch.backends.mps.is_available():
        return ModelDevice.MPS.value
    return ModelDevice.CPU.value
