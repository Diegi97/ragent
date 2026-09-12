import os

import bentoml

from model_services.model_contract import DEFAULT_EMBEDDING_MODEL_NAME
from model_services.runtime import positive_env_int, resolve_device

MODEL_ID = os.getenv("HARRIER_MODEL_ID", DEFAULT_EMBEDDING_MODEL_NAME)


DEVICE = resolve_device("HARRIER_DEVICE")


MAX_SEQ_LENGTH = positive_env_int("HARRIER_MAX_SEQ_LENGTH", 512)


INFERENCE_BATCH_SIZE = positive_env_int("HARRIER_INFERENCE_BATCH_SIZE", 8)


MAX_BATCH_SIZE = positive_env_int("HARRIER_MAX_BATCH_SIZE", 16)


MAX_LATENCY_MS = positive_env_int("HARRIER_MAX_LATENCY_MS", 25)


TIMEOUT_SECONDS = positive_env_int("HARRIER_TIMEOUT_SECONDS", 60)


service_image = bentoml.images.Image(python_version="3.12").python_packages(
    "numpy==2.3.5",
    "sentence-transformers==5.6.0",
    "torch==2.11.0",
    "transformers==5.12.1",
)
