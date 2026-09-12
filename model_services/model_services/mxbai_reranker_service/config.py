import os

import bentoml

from model_services.model_contract import DEFAULT_RERANKER_MODEL_NAME
from model_services.runtime import positive_env_int, resolve_device

MODEL_ID = os.getenv("MXBAI_MODEL_ID", DEFAULT_RERANKER_MODEL_NAME)


DEVICE = resolve_device("MXBAI_DEVICE")


MAX_LENGTH = positive_env_int("MXBAI_MAX_LENGTH", 512)


INFERENCE_BATCH_SIZE = positive_env_int("MXBAI_INFERENCE_BATCH_SIZE", 8)


TIMEOUT_SECONDS = positive_env_int("MXBAI_TIMEOUT_SECONDS", 60)


service_image = bentoml.images.Image(python_version="3.12").python_packages(
    "sentence-transformers==5.6.0",
    "torch==2.11.0",
    "transformers==5.12.1",
)
