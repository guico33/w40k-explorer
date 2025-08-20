# ⚔️ Warhammer 40K Explorer

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenAI](https://img.shields.io/badge/OpenAI-74aa9c?style=for-the-badge&logo=openai&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic-191919?style=for-the-badge&logo=anthropic&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-DC244C?style=for-the-badge&logo=qdrant&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-3771E0?style=for-the-badge&logo=pinecone&logoColor=white)
![SQLite](https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white)

A Retrieval-Augmented Generation (RAG) application that provides an interactive chat interface for exploring Warhammer 40K lore. The system scrapes content from the Warhammer 40K Fandom wiki and uses vector embeddings to enable semantic search and AI-powered responses.

Built with [hexagonal architecture](https://en.wikipedia.org/wiki/Hexagonal_architecture_(software)) for maximum flexibility - supports both OpenAI and Anthropic LLM providers, plus Qdrant and Pinecone vector databases, with easy provider switching via environment variables.

🌐 **Live Application:** [w40k-explorer.streamlit.app](https://w40k-explorer.streamlit.app/)

## Features

- **Interactive Chat Interface**: Streamlit-based chat for asking questions about Warhammer 40K lore
- **Multiple LLM Providers**: Support for OpenAI (GPT-4) and Anthropic (Claude) with easy provider switching
- **Semantic Search**: Vector embeddings enable contextual understanding of queries
- **Citation System**: Answers include numbered citations with links to source material
- **Hexagonal Architecture**: Clean, maintainable code with proper dependency inversion

## Query & Retrieval Strategy

When a user asks a question, the system follows a multi-stage retrieval-augmented generation (RAG) pipeline:

1. **Semantic Retrieval**  
   - The query is embedded and matched against the vector database of Warhammer 40K chunks.  
   - If query expansion is enabled, the engine generates paraphrases and spelling/alias variants (e.g., “Emperor” vs. “God-Emperor”) to improve recall.  
   - Results are ranked by similarity score; if no results are found, the similarity threshold is automatically relaxed.

2. **Result Diversification (MMR-style)**  
   - From the retrieved set, the engine selects diverse passages to avoid redundancy.  
   - A maximum number of chunks per article/section is enforced, with preference for lead paragraphs.  
   - This ensures broad coverage across different sources rather than multiple near-duplicates.

3. **Context Packing**  
   - Selected chunks are truncated to a word limit and enriched with structured metadata (e.g., infobox key-value pairs).  
   - Each context item includes article title, section path, URL, and confidence score.

4. **Answer Generation**  
   - The final context is passed to the LLM with strict instructions:  
     - Answer **only from provided context**.  
     - Every factual claim must cite a passage using `[id]` notation.  
     - Max 10 sentences, concise, with citations.  
     - In case of overly verbose responses, a compressed fallback (≤3 sentences) is attempted.  
   - The LLM outputs a structured JSON object with `answer`, `citations_used`, and a `confidence` score.

5. **Fallback Handling**  
   - If retrieval yields no results → a default “not found” response.  
   - If generation exceeds token limits or fails JSON parsing → retry with tighter constraints or partial recovery.

This design balances **recall** (finding all relevant passages), **precision** (selecting diverse but on-topic chunks), and **trustworthiness** (strict citation rules).

## Quick Start

### Prerequisites

- Python 3.11+
- LLM Provider API key (OpenAI or Anthropic)
- Vector database: Qdrant (cloud or local) or Pinecone

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd w40k-explorer
```

2. Install dependencies:
```bash
run uv sync --all-extras
```

3. Create `.env` file:

```bash
# LLM Provider Selection (choose one)
LLM_PROVIDER=openai  # openai or anthropic

# OpenAI Configuration (required for embeddings, optional for LLM)
OPENAI_API_KEY=your_openai_api_key
OPENAI_LLM_MODEL=gpt-4o-mini-2024-07-18
EMBEDDING_MODEL=text-embedding-3-small

# Anthropic Configuration (only required if LLM_PROVIDER=anthropic)
# ANTHROPIC_API_KEY=your_anthropic_api_key
# ANTHROPIC_LLM_MODEL=claude-3-7-sonnet-latest

# Vector Database Selection (choose one)
VECTOR_PROVIDER=qdrant  # qdrant or pinecone

# Qdrant Configuration (if VECTOR_PROVIDER=qdrant)
QDRANT_URL=https://your-cluster.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key
# Or for local Qdrant:
# QDRANT_HOST=localhost
# QDRANT_PORT=6333

# Pinecone Configuration (if VECTOR_PROVIDER=pinecone)
# PINECONE_API_KEY=your_pinecone_api_key
# PINECONE_INDEX=your_index_name
```

### Running the Application

**Streamlit Chat Interface:**
```bash
uv run streamlit run src/w40k/presentation/streamlit/app.py
```

**Command Line Interface:**
```bash
PYTHONPATH=src uv run python -m w40k.presentation.cli -q "Tell me about the Emperor of Mankind"
```

## Architecture

The application follows **hexagonal architecture** (ports and adapters pattern) for clean separation of concerns:

- **Core Domain**: Business logic and models (`src/w40k/core/`, `src/w40k/usecases/`)
- **Ports**: Interfaces defining contracts (`src/w40k/ports/`)
- **Adapters**: Implementation of external services (`src/w40k/adapters/`)
- **Infrastructure**: Database, vector operations, scrapers (`src/w40k/infrastructure/`)
- **Presentation**: User interfaces (`src/w40k/presentation/`)

### Data Pipeline
**Scrapes** → **Parses** → **Chunks** → **Embeds** → **Stores**

### Key Components
- **LLM Providers**: Pluggable OpenAI/Anthropic clients via adapter pattern
- **Vector Store**: Qdrant or Pinecone for semantic similarity search
- **Query Engine**: RAG pipeline with retrieval, diversification, and generation
- **Database**: SQLite for storage of raw html articles and parsed content

## Project Structure

```
src/w40k/
├── adapters/              # External service implementations
│   ├── llm/              # OpenAI & Anthropic LLM clients
│   └── vector_services/  # Vector service adapters (Qdrant, Pinecone)
├── config/               # Settings and dependency injection
├── core/                 # Domain models and types
├── evals/                # Evaluation framework
│   ├── dataset.py        # Dataset loading and QA item models
│   ├── judges.py         # LLM-based binary judges
│   ├── metrics.py        # Deterministic evaluation metrics
│   ├── report.py         # Result aggregation and reporting
│   └── qa_dataset.jsonl  # W40K evaluation dataset
├── infrastructure/       # Infrastructure services
│   ├── database/         # SQLite operations
│   ├── rag/             # Parsing, chunking, embeddings
│   └── scraper/         # Wiki content scraping  
├── ports/               # Interface definitions
├── presentation/        # User interfaces
│   ├── cli.py          # Command line interface
│   └── streamlit/      # Web interface
└── usecases/           # Business logic
    └── answer.py       # Query processing service

scripts/                # Utility scripts for data processing
├── run_evals.py        # Main evaluation script with parallel processing
```

## Ingestion (Embeddings)

Generate embeddings from the SQLite chunk store and upsert them to the selected vector service.

- Qdrant (local by default):
```bash
uv run python scripts/generate_embeddings.py --db data/articles.db --batch-size 100 --max-chunks 100
```

- Pinecone:
```bash
# In .env set: VECTOR_PROVIDER=pinecone, PINECONE_API_KEY=..., PINECONE_INDEX=...
uv run python scripts/generate_embeddings.py --vector-provider pinecone --db data/articles.db --batch-size 100 --max-chunks 100
```

Flags:
- `--force` recreates the collection/index before syncing
- `--retry-failed` retries any chunks previously marked as failed
- `--dry-run` shows what would be processed without calling APIs

## Evaluation Framework

The project includes a minimal evaluation framework for assessing RAG system performance with both deterministic metrics and LLM-based judges.

### Quick Start

```bash
# Run evaluation on dataset with 5 parallel workers (default)
uv run python scripts/run_evals.py --dataset src/w40k/evals/qa_dataset.jsonl --subset 5 -v

# Custom worker count and vector provider
uv run python scripts/run_evals.py --subset 10 --max-workers 3 --vector-provider qdrant

# Full dataset evaluation  
uv run python scripts/run_evals.py --dataset src/w40k/evals/qa_dataset.jsonl
```

### Environment Variables

```bash
# Evaluation-specific models (optional, falls back to main LLM models)
TEST_MODEL=gpt-4o-mini-2024-07-18     # Model for generating answers
EVAL_MODEL=gpt-4o-mini-2024-07-18     # Model for LLM judges
```

### Metrics

**Deterministic Metrics:**
- **JSON OK**: Valid structured response format
- **Citations Valid**: All citations reference provided context
- **Context Coverage**: Fraction of expected sources found in retrieved context
- **Refusal Rate**: Percentage of safety/refusal responses

**LLM-Based Judges:**
- **Grounded**: Answer supported by provided context (binary judgment)
- **Relevant**: Answer addresses the question appropriately (binary judgment)

**Confidence Calibration:**
- **Expected Calibration Error (ECE)**: Measures alignment between predicted confidence and actual correctness

### Dataset Format

Questions are stored as JSONL with the following structure:
```json
{
  "id": "unique-identifier",
  "question": "What is the Emperor of Mankind?",
  "expected_sources": ["https://warhammer40k.fandom.com/wiki/Emperor_of_Mankind"]
}
```

### Output Structure

Each evaluation run creates a timestamped directory in `evals/runs/` containing:
- `report.md`: Summary metrics and statistics
- `samples.jsonl`: Detailed per-sample results
- `summary.json`: Aggregated metrics in JSON format
- `meta.json`: Run configuration and metadata
- `run.log`: Detailed execution logs

### Parallel Processing

The framework supports parallel evaluation with configurable concurrency:
- `--max-workers N`: Set maximum concurrent evaluations (default: 5)
- Optimized for I/O-bound operations (API calls)
- Real-time progress tracking and result streaming
- Thread-safe result aggregation

## Testing

The test suite currently focuses on the inference path.

- Philosophy: deterministic, no network, tests behavior (no implementation details).
- What's covered:
  - AnswerService: JSON parsing, citation construction, truncation retry, threshold relaxation, refusal handling, confidence clamping, MMR de‑dup.
  - Adapters (contracts): OpenAI and Anthropic return "Responses‑like" objects `status/output/message/output_text` that the use case parses.
  - Streamlit utils: citation remapping `[ID]` → sequential `[1]`, ordered sources.
  - Config/Factory: provider selection and embedding preconditions validated.

### Run tests

```bash
# Full test suite (fast)
uv run pytest -q

# Only inference use cases
uv run pytest -q tests/usecases

# Adapter contract tests
uv run pytest -q tests/adapters

# With coverage
uv run pytest --cov=w40k --cov-report=html
```

### Test layout

```
tests/
├── adapters/                 # Contract tests for LLM adapters
├── config/                   # Factory/config validation tests
├── fakes/                    # Deterministic doubles (LLM, VectorService)
├── presentation/             # Streamlit utils tests
└── usecases/                 # AnswerService behavior
```

Notes:
- No API keys or external services are required — fakes are used throughout.
- `tests/conftest.py` adds `src/` to `PYTHONPATH` for clean imports.
- `pytest.ini` sets sensible defaults (strict markers, short tracebacks).
