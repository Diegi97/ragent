# Format code with Black
format_black:
	@echo "Formatting code with Black..."
	black ./ragent
	@echo "Black formatting completed"

# Sort imports with isort
format_isort:
	@echo "Sorting imports with isort..."
	isort ./ragent
	@echo "Import sorting completed"

# Run all formatters
format_code: format_black format_isort
	@echo "All code formatting completed!"

populate_db:
	python -c 'from ragent.data.sources.bm25s import BM25Client; BM25Client.load_retriever()'

.PHONY: default load_people_fixtures load_deployedapps_fixtures load_fixtures reset_db reset_and_load_fixtures format_black format_isort format_code