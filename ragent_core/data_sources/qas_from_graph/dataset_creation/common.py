import json
import os

from ragent_core.data_sources.qas_from_graph.graph import GraphStore, GraphExpander, MaxRandomSearchReached

LIGHT_SAVE = True
MAX_TRIES_PER_ITEM = 5

expander = GraphExpander()

def create_dataset_item(width: int, depth: int) -> tuple[bool, tuple[GraphStore | None, str | None, int, int]]:
    graph_store = GraphStore()
    expander.init_graph(graph_store, force=True)

    for _ in range(MAX_TRIES_PER_ITEM):
        answer_node, answer_attrs = graph_store.get_random_node()

        width_expand_success = expander.width_expand_until_single_result(graph_store, answer_node, width=width)
        if width_expand_success:
            depth_expand_success = expander.depth_expand(graph_store, answer_node, depth=depth)
            if depth_expand_success:
                return True, (graph_store, answer_node, width, depth)
    return False, (None, None, width, depth)

def add_result_when_done(dataset, success, result, pbar):
    if success:
        dataset.append({
            "graph_store": result[0].to_dict(light=LIGHT_SAVE),
            "answer_node": result[1],
            "width": result[2],
            "depth": result[3],
        })
        pbar.update(1)


def depth_width_generator(min_width: int, max_width: int, min_depth: int, max_depth: int):
    while True:
        for width in range(min_width, max_width + 1):
            for depth in range(min_depth, max_depth + 1):
                yield width, depth


def save_dataset(path: str, dataset: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(dataset, f, indent=4)
