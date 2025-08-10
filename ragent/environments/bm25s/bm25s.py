from datasets import load_dataset
import verifiers as vf

from ragent.data.sources.bm25s import BM25Client
from ragent.data.pipelines import safe_ds_name

def load_environment(**kwargs) -> vf.Environment:
    hf_ds = kwargs["hf_ds"]

    bm25_client = BM25Client(hf_ds)
    qas_dataset = load_dataset(f"data/{safe_ds_name(hf_ds)}/", data_files="qas.json")

    return vf.ToolEnv(
        dataset=qas_dataset,
        # rubric= ... # Rubric object; vf.ToolRubric() can be optionally used for counting tool invocations in each rollout
        tools=[bm25_client.search_tool, bm25_client.read_tool], # python functions with type hints + docstrings
        max_turns=10
    )
