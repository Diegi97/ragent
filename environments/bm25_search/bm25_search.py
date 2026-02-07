from __future__ import annotations

import logging
from typing import Mapping

import verifiers as vf
from datasets import Dataset, DatasetDict, load_dataset
from ragent_core.prompts.agent.search_engine import PROMPT as AGENT_PROMPT
from ragent_core.retrievers.bm25_retriever import BM25Retriever
from ragent_core.rewards import format_reward, judge_reward

logger = logging.getLogger(__name__)


JUDGE_REWARD_FUNC_WEIGHT = 0.8
GET_FORMAT_REWARD_FUNC_WEIGHT = 0.2


class MultiSourceBM25Environment(vf.StatefulToolEnv):
    def __init__(self, data_sources, **kwargs):
        super().__init__(**kwargs)
        self._build_indexes(data_sources)
        self.add_tool(self.search_tool, args_to_skip=["data_source"])
        self.add_tool(self.read_tool, args_to_skip=["data_source"])
        self.add_tool(self.text_scan_tool, args_to_skip=["data_source"])

    def _build_indexes(
        self,
        datasets: Mapping[str, Dataset] | DatasetDict,
        **kwargs,
    ) -> None:
        self.datasets: dict[str, Dataset] = dict(datasets)
        self._retrievers: dict[str, BM25Retriever] = {}

        for source_name, dataset in self.datasets.items():
            self._retrievers[source_name] = BM25Retriever(
                dataset=dataset,
                dataset_name=source_name,
                **kwargs,
            )

        logger.info(
            "Initialized MultiSourceBM25Retriever with sources: %s",
            ", ".join(sorted(self._retrievers.keys())),
        )

    def update_tool_args(self, tool_name, tool_args, messages, state, **kwargs):
        tool_args["data_source"] = state.get("input").get("info").get("data_source")
        return tool_args

    def search_tool(self, queries: list[str], data_source: str) -> str:
        return self._retrievers[data_source].search_tool(queries)

    def read_tool(self, doc_ids: list[int], data_source: str) -> str:
        return self._retrievers[data_source].read_tool(doc_ids)

    def text_scan_tool(
        self,
        pattern: str,
        data_source: str,
        fixed_string: bool = True,
        case_sensitive: bool = False,
        max_results: int = 25,
        snippet_chars: int = 200,
    ) -> str:
        return self._retrievers[data_source].text_scan_tool(
            pattern, fixed_string, case_sensitive, max_results, snippet_chars
        )


def load_environment() -> vf.Environment:
    """
    Construct the BM25 evaluation environment.

    Args:
        bm25_dataset: HF dataset identifier used by the BM25 client.
        qas_dataset: Q&A dataset location. Accepts either a HF dataset ID or a local path. Local path for flexibility while developing.
    """
    dataset = load_dataset("diegi97/ragent_qa_pairs")

    def transform_prompt(example):
        """Transform prompt column from string to list of message dicts."""
        return {"prompt": [{"role": "user", "content": example["prompt"]}]}

    dataset = dataset.map(transform_prompt)
    data_sources = load_dataset("diegi97/ragent_data_sources")

    rubric = vf.Rubric(
        funcs=[judge_reward, format_reward],
        weights=[JUDGE_REWARD_FUNC_WEIGHT, GET_FORMAT_REWARD_FUNC_WEIGHT],
    )

    return MultiSourceBM25Environment(
        data_sources=data_sources,
        dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        rubric=rubric,
        system_prompt=AGENT_PROMPT,
        max_turns=30,
    )
