# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Philosophy

**Keep implementations minimal and straightforward. Avoid over-engineering.**

- Prefer simple, direct solutions over complex abstractions
- Choose readability over cleverness 
- Implement only what's needed, when it's needed
- Avoid premature optimization and unnecessary features
- Use standard library and well-established patterns

## Git Commit Guidelines

- Write clear, concise commit messages focused on the change
- Never mention Claude Code, Claude, or AI assistance in commit messages
- Focus on what was changed and why, not how it was created
- Use conventional commit format when appropriate

## Project Overview

This is a Warhammer 40k RAG (Retrieval-Augmented Generation) application that scrapes content from the Warhammer 40k Fandom wiki and provides a Streamlit chat interface for querying lore. The project is designed for deployment on Streamlit Cloud.

## Architecture

The application follows a multi-phase architecture:

1. **Data Collection**: Web scraper stores raw HTML in SQLite with compression
2. **Data Processing**: Content extraction, chunking, and embedding generation 
3. **Vector Storage**: Qdrant database for semantic search
4. **RAG Pipeline**: Retrieval + GPT-4o-mini generation with citations
5. **Frontend**: Streamlit chat interface with conversation memory

## Tech Stack

- **Storage**: SQLite (raw data) → Qdrant (vectors)
- **Processing**: BeautifulSoup → LangChain
- **LLM**: GPT-4o-mini (OpenAI)
- **Embeddings**: text-embedding-3-small (OpenAI)
- **Framework**: LangChain for RAG orchestration
- **UI**: Streamlit
- **Deployment**: Streamlit Cloud + Docker

## Key Implementation Details

- Target: ~7,000 articles from warhammer40k.fandom.com
- Chunking: ~500 tokens with context preservation
- Caching: SQLite for LLM responses and semantic similarity
- Rate limiting with exponential backoff for scraping
- Citation tracking to source articles

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Install with development dependencies
uv sync --extra dev

# Install with optional enhancements (cohere, networkx)
uv sync --extra enhancements
```

### Code Quality
```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking
uv run mypy src/

# Run all quality checks
uv run ruff check . && uv run ruff format . && uv run mypy src/
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test file
uv run pytest tests/unit/test_scraper.py
```

### Development Server
```bash
# Run Streamlit app
uv run streamlit run src/ui/main.py

# Run with specific port
uv run streamlit run src/ui/main.py --server.port 8502
```

### Project Structure
```
src/
├── scraper/     # Web scraping functionality
├── database/    # SQLite operations and schema
├── rag/         # RAG pipeline and embeddings
└── ui/          # Streamlit interface
tests/           # Test files
data/            # Local data storage
docs/            # Documentation
```

## Performance Targets

- Response time: <3 seconds
- Retrieval accuracy: 90%+ relevance
- Operational costs: <$10/month
- Data coverage: 7,000+ indexed articles