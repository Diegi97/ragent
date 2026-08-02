from typing import List, Optional

from ragent_core.retrievers.document import RetrievalResult
from ragent_core.retrievers.mode import RetrievalMode


class BaseRetriever:
    """Minimal contract for retrievers.

    A retriever takes a list of :class:`Document` records at construction
    time, indexes them, and returns the top-ranked ones for a query. It does
    not know anything about chunking or full corpora -- those concepts belong
    to :class:`AgentRetriever`.
    """

    def retrieve(
        self,
        query: str,
        table_name: str,
        top_k: int = 50,
        retrieval_mode: Optional[RetrievalMode] = None,
        **kwargs,
    ) -> List[RetrievalResult]:
        raise NotImplementedError
