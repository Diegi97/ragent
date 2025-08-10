import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGING_FACE_DATASET = os.getenv("HUGGING_FACE_DATASET")
