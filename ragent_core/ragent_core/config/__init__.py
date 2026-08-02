import os

from dotenv import load_dotenv

from .logging import configure_logging

load_dotenv()

configure_logging()

HF_TOKEN = os.getenv("HF_TOKEN")
LANCEDB_GCS_URI = os.getenv("LANCEDB_GCS_URI")
GCS_SERVICE_ACCOUNT = os.getenv("GCS_SERVICE_ACCOUNT")
