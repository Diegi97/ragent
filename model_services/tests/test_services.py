import unittest

import numpy as np
import torch

from model_services.harrier_service import (
    MAX_BATCH_SIZE,
    MAX_LATENCY_MS,
    QUERY_PROMPT_NAME,
    HarrierEmbeddingService,
    HarrierEncoder,
)
from model_services.mxbai_reranker_service import (
    INFERENCE_BATCH_SIZE,
    MxbaiRanker,
    MxbaiRerankerService,
)


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.calls = []
        self.max_seq_length = None

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return np.array(
            [[index + 0.123456789, index + 1.123456789] for index in range(len(texts))],
            dtype=np.float64,
        )


class FakeCrossEncoder:
    def __init__(self) -> None:
        self.calls = []

    def rank(self, query, texts, **kwargs):
        self.calls.append((query, texts, kwargs))
        return [
            {"corpus_id": index, "score": float(len(texts) - index)}
            for index in range(len(texts))
        ]


class HarrierServiceTest(unittest.TestCase):
    def test_query_and_document_prompt_contracts(self) -> None:
        model = FakeEmbeddingModel()
        encoder = HarrierEncoder(model=model)

        query_embeddings = encoder.encode_queries(["query"])
        document_embeddings = encoder.encode_documents(["document"])

        self.assertEqual(model.calls[0][1]["prompt_name"], QUERY_PROMPT_NAME)
        self.assertNotIn("prompt_name", model.calls[1][1])
        self.assertTrue(model.calls[0][1]["normalize_embeddings"])
        self.assertTrue(model.calls[1][1]["normalize_embeddings"])
        self.assertEqual(
            query_embeddings,
            np.asarray([[0.123456789, 1.123456789]], dtype=np.float32).tolist(),
        )
        self.assertEqual(len(document_embeddings), 1)

    def test_embedding_apis_use_adaptive_batching(self) -> None:
        for api_name in ("encode_queries", "encode_documents"):
            api = HarrierEmbeddingService.apis[api_name]
            self.assertTrue(api.batchable)
            self.assertEqual(api.max_batch_size, MAX_BATCH_SIZE)
            self.assertEqual(api.max_latency_ms, MAX_LATENCY_MS)


class MxbaiServiceTest(unittest.TestCase):
    def test_rank_returns_raw_sorted_scores(self) -> None:
        model = FakeCrossEncoder()
        ranker = MxbaiRanker(model=model)

        results = ranker.rank("query", ["first", "second"])

        self.assertEqual(
            results,
            [
                {"corpus_id": 0, "score": 2.0},
                {"corpus_id": 1, "score": 1.0},
            ],
        )
        kwargs = model.calls[0][2]
        self.assertEqual(kwargs["batch_size"], INFERENCE_BATCH_SIZE)
        self.assertIsInstance(kwargs["activation_fn"], torch.nn.Identity)

    def test_rank_empty_input_does_not_call_model(self) -> None:
        model = FakeCrossEncoder()
        ranker = MxbaiRanker(model=model)
        self.assertEqual(ranker.rank("query", []), [])
        self.assertEqual(model.calls, [])

    def test_reranker_api_is_not_batchable(self) -> None:
        self.assertFalse(MxbaiRerankerService.apis["rerank"].batchable)


if __name__ == "__main__":
    unittest.main()
