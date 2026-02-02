import os
import time
import traceback

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from ragent_core.data_sources.qas_from_graph.dataset_creation.common import save_dataset, create_dataset_item, \
    add_result_when_done, depth_width_generator

MAX_THREADS = 20

def wait_for_any_to_complete(running_futures, dataset, pbar):
    done_now = {}
    while not done_now:
        done_now = {f for f in running_futures if f.done()}
        if not done_now:
            time.sleep(0.01)
            continue
        for f in done_now:
            try:
                success, result = f.result()
            except Exception:
                success, result = False, None
            running_futures.remove(f)
            add_result_when_done(dataset, success, result, pbar)

def create_dataset_multi_thread(dataset_size: int, min_width: int, max_width: int, min_depth: int, max_depth: int) -> list[dict]:
    dataset = []
    running_futures = set()
    depth_width_gen = depth_width_generator(min_width, max_width, min_depth, max_depth)

    try:
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            with tqdm(total=dataset_size) as pbar:
                while len(dataset) < dataset_size:

                    if len(running_futures) >= MAX_THREADS:
                        wait_for_any_to_complete(running_futures, dataset, pbar)

                    width, depth = next(depth_width_gen)
                    f = executor.submit(create_dataset_item, width, depth)
                    running_futures.add(f)

                while running_futures:
                    wait_for_any_to_complete(running_futures, dataset, pbar)
    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            print("\n⛔ Keyboard interrupt detected. Shutting down threads...")
        else:
            print("Shutting down because of error...")
            traceback.print_exc()
        for f in running_futures: f.cancel()
    return dataset

if __name__ == '__main__':
    DATASET_SIZE = 10000
    MAX_WIDTH = 10
    MIN_WIDTH = 2
    MAX_DEPTH = 10
    MIN_DEPTH = 2


    dataset = create_dataset_multi_thread(DATASET_SIZE, MIN_WIDTH, MAX_WIDTH, MIN_DEPTH, MAX_DEPTH)
    PATH_RESULTS = os.path.join(os.path.dirname(__file__), 'results', f'dataset_{len(dataset)}_W{MIN_WIDTH}x{MAX_WIDTH}_D{MIN_DEPTH}x{MAX_DEPTH}.json')
    save_dataset(PATH_RESULTS, dataset)
    print(f"✅ Dataset saved to {PATH_RESULTS}, total items: {len(dataset)}")
