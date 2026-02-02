import importlib


def safe_ds_name(dataset_name):
    return dataset_name.replace("-", "_").replace("/", "_").replace(".", "_")


def get_pipeline_run(dataset_name):
    module_name = safe_ds_name(dataset_name)
    module = importlib.import_module(f".{module_name}", package=__package__)
    return getattr(module, "run")
