import os

PROJECT_NAME = "deep-search-task-generation"
LLM_CONCURRENCY_LIMIT = "deep-search-tasks-openai-llm"


def phoenix_project() -> str:
    return os.getenv("PHOENIX_DEEP_SEARCH_TASK_GENERATION_PROJECT_NAME", PROJECT_NAME)
