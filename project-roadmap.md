# Warhammer 40k RAG Application - Project Roadmap

## Project Overview
Build a RAG (Retrieval-Augmented Generation) application for Warhammer 40k lore using scraped content from the Warhammer 40k Fandom wiki, with a Streamlit chat interface and deployed on Streamlit Cloud.

## Complete Project Roadmap

### Phase 1: Data Collection & Storage

#### 1. SQLite Persistence Layer
- Implement database schema for raw HTML storage with compression
- Add metadata tables and indexing
- **Technologies:** `sqlite3`, `zlib` for compression

#### 2. Web Scraper Implementation
- Build scraper with BeautifulSoup for parsing
- Implement async fetching with rate limiting
- Add progress tracking and resume capability
- **Technologies:** `beautifulsoup4`, `aiohttp`, `tqdm`

#### 3. Execute Scraping
- Fetch all ~7k articles from warhammer40k.fandom.com
- Store raw HTML in SQLite
- **Estimated time:** 6-12 hours runtime

#### 4. Data Validation & QA
- Check for failed scrapes and retry logic
- Identify and remove duplicates
- Validate HTML integrity
- Generate scraping report/statistics

### Phase 2: Data Processing & Indexing

#### 5. Parse & Extract Structured Content
- Extract article text, infoboxes, categories
- Build metadata (links, sections, character mentions)
- Clean and normalize text
- **Technologies:** `beautifulsoup4`, `regex`

#### 6. Document Loading & Chunking
- Implement LangChain document loaders from SQLite
- Smart chunking strategy (preserve context, ~500 tokens)
- Add chunk metadata (source, section, URL)
- **Technologies:** `langchain`, `tiktoken`

#### 7. Generate Embeddings & Vector Storage
- Create embeddings using OpenAI's text-embedding-3-small
- Store in Qdrant vector database
- Implement batch processing for efficiency
- **Technologies:** `openai`, `qdrant-client`, `langchain`

### Phase 3: RAG Implementation

#### 8. Evaluation Harness
- Create test queries (simple facts, complex lore, relationships)
- Validate retrieval quality
- Tune chunk size and retrieval parameters
- Measure response accuracy

#### 9. RAG Query Pipeline
- Implement retrieval with Qdrant
- Add reranking layer (Cohere or cross-encoder)
- Integrate GPT-4o-mini for response generation
- Include citation tracking to source articles
- **Technologies:** `langchain`, `openai`, `cohere` (optional)

#### 10. Caching Layer
- Implement SQLite cache for LLM responses
- Add semantic cache for similar queries
- Track token usage and costs
- **Technologies:** `langchain.cache`, `sqlite3`

### Phase 4: Application Development

#### 11. Streamlit Chat Interface
- Build conversational UI with message history
- Add conversation memory for follow-up questions
- Display citations with links to wiki articles
- Include example queries and help section
- **Technologies:** `streamlit`, `streamlit-chat`

### Phase 5: Deployment & Monitoring

#### 12. Docker Containerization
- Create Dockerfile with all dependencies
- Use multi-stage build to optimize size
- Add docker-compose for local development
- **Technologies:** Docker

#### 13. Streamlit Cloud Deployment
- Configure secrets management (API keys)
- Set up requirements.txt and packages.txt
- Configure resource limits
- Implement health checks

#### 14. Monitoring & Analytics
- Track usage metrics (queries/day, popular topics)
- Monitor costs (OpenAI API usage)
- Log errors and failed queries
- User feedback collection
- **Technologies:** `streamlit-analytics`, custom logging

## Tech Stack Summary

### Core Technologies
- **Storage:** SQLite (raw data) → Qdrant (vectors)
- **Processing:** BeautifulSoup → LangChain
- **LLM:** GPT-4o-mini (OpenAI)
- **Embeddings:** text-embedding-3-small (OpenAI)  
- **Framework:** LangChain for RAG orchestration
- **UI:** Streamlit
- **Deployment:** Streamlit Cloud
- **Containerization:** Docker

### Python Dependencies

```python
# Core
beautifulsoup4
aiohttp
sqlite3  # built-in
langchain
langchain-openai
qdrant-client
openai

# UI
streamlit
streamlit-chat

# Utilities
tqdm
tiktoken
pandas
python-dotenv

# Optional enhancements
cohere  # for reranking
networkx  # for knowledge graphs
```

## Key Metrics for Success

- **Data Coverage:** 7,000+ articles successfully scraped and indexed
- **Response Time:** <3 seconds for typical queries
- **Accuracy:** 90%+ relevance in retrieved chunks
- **Cost Efficiency:** <$10/month operational costs
- **User Experience:** Smooth chat interface with clear citations

## Risk Mitigation

- **Rate Limiting:** Implement exponential backoff for scraping
- **Cost Overruns:** Cache frequently asked queries, monitor API usage
- **Data Quality:** Validate scraped content, implement fallbacks
- **Deployment Issues:** Test locally with Docker before cloud deployment