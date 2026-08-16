import os

from dotenv import load_dotenv

from .logging import configure_logging

load_dotenv()

configure_logging()

HF_TOKEN = os.getenv("HF_TOKEN")
