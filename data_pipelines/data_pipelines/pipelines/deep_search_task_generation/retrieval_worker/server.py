import logging
import os
import signal
from typing import Any

from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker.broker import (
    RetrieverBroker,
)
from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker.config import (
    RetrieverWorkerConfig,
)
from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker.contracts import (
    RETRIEVER_WORKER_HOST,
    authkey_from_environment,
)
from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker.transport import (
    RetrieverManager,
)
from ragent_core.retrievers.retriever import TurbopufferRetriever
from ragent_core.retrievers.settings import DEFAULT_RERANK_BATCH_SIZE

logger = logging.getLogger(__name__)


def load_worker_retriever(config: RetrieverWorkerConfig) -> TurbopufferRetriever:
    return TurbopufferRetriever.load_index(
        namespace=config.retriever_namespace,
        device=config.retriever_device,
        rerank_threshold=config.rerank_threshold,
        top_rerank=1,
        rerank_batch_size=DEFAULT_RERANK_BATCH_SIZE,
    )


def serve_retriever_worker(config: RetrieverWorkerConfig) -> None:
    authkey = authkey_from_environment()
    retriever = load_worker_retriever(config)
    broker = RetrieverBroker(retriever, config)
    manager = RetrieverManager.serving(
        broker,
        address=(RETRIEVER_WORKER_HOST, config.port),
        authkey=authkey,
    )
    server = manager.get_server()

    def stop_on_sigterm(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, stop_on_sigterm)
    broker.start()
    logger.info(
        "Retriever worker ready on %s:%d (pid=%d)",
        RETRIEVER_WORKER_HOST,
        config.port,
        os.getpid(),
    )
    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Stopping retriever worker after the active request finishes.")
    finally:
        broker.stop()
        signal.signal(signal.SIGTERM, previous_sigterm)
