from collections.abc import Sequence

import numpy as np

from ragent_core.retrievers.model_contract import (
    EMBEDDING_DOCUMENTS_ROUTE,
    EMBEDDING_QUERY_ROUTE,
    normalize_embeddings,
)
from ragent_core.retrievers.service_clients.transport import ModelServiceClient


class EmbeddingServiceClient(ModelServiceClient):
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(EMBEDDING_QUERY_ROUTE, texts)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(EMBEDDING_DOCUMENTS_ROUTE, texts)

    def _encode(self, path: str, texts: Sequence[str]) -> np.ndarray:
        normalized_texts = [str(text) for text in texts]
        if not normalized_texts:
            return np.empty((0, 0), dtype=np.float32)
        payload = self.post(path, {"texts": normalized_texts})
        return normalize_embeddings(payload, len(normalized_texts))
