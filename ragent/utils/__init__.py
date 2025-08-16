import os

from dotenv import load_dotenv

from ragent.utils.logging import configure_logging
from ragent.utils.config import TrainConfig, prepare_args

load_dotenv()

configure_logging()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
