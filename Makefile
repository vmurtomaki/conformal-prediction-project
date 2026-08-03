.PHONY: setup test lint format

setup:
	git init
	uv sync --all-extras --dev

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ app/ tests/
	uv run mypy src/ app/

format:
	uv run ruff format src/ app/ tests/