# ⚔️ Warhammer 40K Knowledge Base

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenAI](https://img.shields.io/badge/OpenAI-74aa9c?style=for-the-badge&logo=openai&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-DC244C?style=for-the-badge&logo=qdrant&logoColor=white)
![SQLite](https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white)

A Retrieval-Augmented Generation (RAG) application that provides an interactive chat interface for exploring Warhammer 40K lore. The system scrapes content from the Warhammer 40K Fandom wiki and uses vector embeddings to enable semantic search and AI-powered responses.

🌐 **Live Application:** [w40k-explorer.streamlit.app](https://w40k-explorer.streamlit.app/)

## Features

- **Interactive Chat Interface**: Streamlit-based chat for asking questions about Warhammer 40K lore
- **Semantic Search**: Vector embeddings enable contextual understanding of queries
- **Citation System**: Answers include numbered citations with links to source material

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

- Python 3.8+
- OpenAI API key
- Remote Qdrant vector database 

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd w40k-explorer
```

2. Install dependencies:
```bash
uv sync 
```

3. Create `.env` file:
```bash
# Required
OPENAI_API_KEY=your_openai_api_key
OPENAI_LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Optional (for Qdrant Cloud)
QDRANT_URL=https://your-cluster.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key
```

### Running the Application

**Streamlit Chat Interface:**
```bash
uv run streamlit run src/ui/streamlit_app.py
```

## Architecture

- **Data Pipeline**: Scrapes > Parses > Chunks > Embeds > Stores
- **Query Engine**: Retrieves relevant chunks and generates contextual responses
- **Vector Store**: Qdrant to store embeddings and chunks data for semantic similarity search
- **Database**: SQLite to store articles and chunks prior to embeddings generation

## Project Structure

```
src/
    scraper/       # Wiki content scraping
    database/      # SQLite operations and vector management
    rag/           # Parsing, chunking, and embeddings
    engine/        # Query processing and response generation
    ui/            # Streamlit chat interface
```

See `CLAUDE.md` for development guidelines.