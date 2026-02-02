import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .logging import configure_logging
from .train import TrainConfig, prepare_args

load_dotenv()

configure_logging()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = os.getenv("OPENROUTER_URL")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
JUDGE_MODEL = os.getenv("JUDGE_MODEL")
JUDGE_CLIENT = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_URL
)
