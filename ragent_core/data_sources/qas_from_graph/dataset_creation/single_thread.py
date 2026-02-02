import os
import traceback

from tqdm import tqdm

from ragent_core.data_sources.qas_from_graph.graph import GraphStore
from ragent_core.data_sources.qas_from_graph.dataset_creation.common import depth_width_generator, create_dataset_item, \
    add_result_when_done, save_dataset


def create_dataset(dataset_size: int, min_width: int, max_width: int, min_depth: int, max_depth: int) -> list[tuple[GraphStore, str]]:
    dataset = []
    depth_width_gen = depth_width_generator(min_width, max_width, min_depth, max_depth)

    with tqdm(total=dataset_size) as pbar:
        try:
            while len(dataset) < dataset_size:
                width, depth = next(depth_width_gen)
                try:
                    success, result = create_dataset_item(width, depth)
                except Exception:
                    success = False
                    result = None
                add_result_when_done(dataset, success, result, pbar)

        except (KeyboardInterrupt, Exception) as e:
            if isinstance(e, KeyboardInterrupt):
                print("\n⛔ Keyboard interrupt detected. Shutting down threads...")
            else:
                print("Shutting down because of error...")
    return dataset


if __name__ == '__main__':
    DATASET_SIZE = 10000
    MAX_WIDTH = 10
    MIN_WIDTH = 2
    MAX_DEPTH = 10
    MIN_DEPTH = 2


    dataset = create_dataset(DATASET_SIZE, MIN_WIDTH, MAX_WIDTH, MIN_DEPTH, MAX_DEPTH)
    PATH_RESULTS = os.path.join(os.path.dirname(__file__), 'results', f'dataset_{len(dataset)}_W{MIN_WIDTH}x{MAX_WIDTH}_D{MIN_DEPTH}x{MAX_DEPTH}.json')
    save_dataset(PATH_RESULTS, dataset)
    print(f"✅ Dataset saved to {PATH_RESULTS}, total items: {len(dataset)}")
