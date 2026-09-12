import bentoml

from model_services.model_contract import RERANK_ROUTE
from model_services.mxbai_reranker_service.config import TIMEOUT_SECONDS, service_image
from model_services.mxbai_reranker_service.model import MxbaiRanker


@bentoml.service(
    name="ragent-mxbai-reranker",
    image=service_image,
    workers=1,
    traffic={"timeout": TIMEOUT_SECONDS},
)
class MxbaiRerankerService:
    def __init__(self) -> None:
        self.ranker = MxbaiRanker()

    @bentoml.api(route=RERANK_ROUTE)
    def rerank(
        self,
        query: str,
        texts: list[str],
    ) -> list[dict[str, int | float]]:
        return self.ranker.rank(query, texts)
