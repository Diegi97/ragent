import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from model_services import model_contract
from model_services.harrier_service import HarrierEmbeddingService
from model_services.mxbai_reranker_service import MxbaiRerankerService


def core_module(name):
    path = (
        Path(__file__).resolve().parents[2]
        / f"ragent_core/ragent_core/retrievers/{name}.py"
    )
    spec = importlib.util.spec_from_file_location(f"core_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_client_and_service_contracts_agree():
    core = core_module("model_contract")
    for name in (
        "EMBEDDING_QUERY_ROUTE",
        "EMBEDDING_DOCUMENTS_ROUTE",
        "RERANK_ROUTE",
        "QUERY_PROMPT_NAME",
    ):
        assert getattr(core, name) == getattr(model_contract, name)
    settings = core_module("settings")
    for name in ("DEFAULT_EMBEDDING_MODEL_NAME", "DEFAULT_RERANKER_MODEL_NAME"):
        assert getattr(settings, name) == getattr(model_contract, name)
    assert (
        HarrierEmbeddingService.apis["encode_queries"].route
        == core.EMBEDDING_QUERY_ROUTE
    )
    assert (
        HarrierEmbeddingService.apis["encode_documents"].route
        == core.EMBEDDING_DOCUMENTS_ROUTE
    )
    assert MxbaiRerankerService.apis["rerank"].route == core.RERANK_ROUTE
    results = [model_contract.RankResult(corpus_id=0, score=2.0)]
    assert core.normalize_rank_results(
        results, 1
    ) == model_contract.normalize_rank_results(results, 1)


@pytest.mark.parametrize("payload", [[[]], [[float("nan")]], [[1], [2]]])
def test_independent_adapters_reject_the_same_invalid_embedding_shapes(payload):
    for contract in (core_module("model_contract"), model_contract):
        with pytest.raises(contract.ModelServiceError):
            contract.normalize_embeddings(payload, 1)


def test_generated_service_contract_is_current():
    script = Path(__file__).resolve().parents[1] / "scripts/sync_model_contract.py"
    subprocess.run([sys.executable, str(script), "--check"], check=True)
