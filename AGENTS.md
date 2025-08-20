# Repository Guidelines

This guide helps contributors work effectively in w40k-explorer.

## Project Structure & Modules
- `src/`: Application code
  - `scraper/`: URL collection and HTML fetching
  - `rag/`: Chunking, embeddings, and vector store integration
  - `database/`: SQLite models, sessions, vector ops
  - `engine/`: Query engine and helpers
  - `ui/`: Placeholder for Streamlit
- `scripts/`: CLI utilities (scrape, parse, chunk, embed, query)
- `migrations/`: SQLite schema migration helpers
- `data/`: Local artifacts (e.g., `articles.db`)
- `tests/`: Pytest home (currently minimal)
- `.env.example`: Required configuration template

## Build, Test, and Dev Commands
Use uv (preferred) or pip.
- Setup (uv): `uv sync --all-extras`
- Test (if added): `uv run pytest -q`
- Run query CLI: `uv run python scripts/test_query_cli.py --db data/articles.db --query "Who are the Ultramarines?"`
- Generate embeddings (dry run): `uv run python scripts/generate_embeddings.py --db data/articles.db --dry-run`
- Apply migration (preview): `uv run python migrations/2025_08_17_add_embedding_tracking.py data/articles.db --dry-run`

## Coding Style & Naming
- Naming: `snake_case` for modules/functions, `CamelCase` for classes, `UPPER_SNAKE` for constants.
- Keep functions small; prefer pure helpers in `rag/` and `engine/`.

## Testing Guidelines
- Framework: `pytest` (and `pytest-asyncio` when needed).
- Location: `tests/`
- Conventions: name tests `test_*.py`, arrange Given/When/Then, avoid network by default; mark integration tests that require OpenAI/LLM or the vector service and guard with env checks.

## Commit & Pull Request Guidelines
- Commits: Imperative, concise, scoped (e.g., "Implement structure-aware chunking pipeline").
- PRs must include:
  - Clear summary and motivation; link issues
  - Usage notes (e.g., commands to run) and screenshots/logs if UI/CLI behavior changes
  - Checklist: lint, format, type-check, tests passing

## Security & Config Tips
- Copy `.env.example` to `.env` and set: `OPENAI_API_KEY`, `OPENAI_LLM_MODEL`, `EMBEDDING_MODEL`; optionally `QDRANT_URL`, `QDRANT_API_KEY`.
- Do not commit secrets or `data/` artifacts. Keep queries deterministic in tests by mocking LLM and vector service adapters.
