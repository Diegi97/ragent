"""Run search-query commands with python -m data_pipelines.cli."""

if __name__ == "__main__":
    from data_pipelines.pipelines.search_query_generation.cli import app

    app()
