import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGING_FACE_DATASET = os.getenv("HUGGING_FACE_DATASET")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
