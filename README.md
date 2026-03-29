# 🧠 NeuroRAG — Neurodivergent Information Retrieval System

A full end-to-end RAG (Retrieval-Augmented Generation) pipeline that scrapes,
embeds, indexes, and semantically searches scientific literature on neurodivergent
conditions (ADHD, Autism, Dyslexia, Dyspraxia, and more).

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Data Pipeline                       │
│  PubMed ──┐                                             │
│  ArXiv  ──┼──► Scraper ──► Embedding Engine ──► VectorDB│
│  MedRxiv──┘    (unified)   (sentence-transformers)      │
│  S2     ──┘                                             │
└─────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼──────────────────────┐
              │ FAISS (local dev)   │   Pinecone (cloud)   │
              └─────────────────────┴──────────────────────┘
                                    │
                           FastAPI backend
                                    │
                          Streamlit frontend
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/neuro-rag.git
cd neuro-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set PUBMED_EMAIL
```

### 3. Build the index (scrape + embed + index)

```bash
python scripts/build_index.py --max-results 100
```

This will:
- Scrape articles from PubMed, ArXiv, MedRxiv, and Semantic Scholar
- Embed them with `all-MiniLM-L6-v2`
- Save a FAISS index to `./data/faiss_index`

### 4. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Start the frontend

```bash
streamlit run frontend/app.py
```

Visit `http://localhost:8501` 🎉

---

## Docker (full stack)

```bash
cp .env.example .env   # fill in your values
docker-compose up --build
```

| Service  | URL                    |
|----------|------------------------|
| API      | http://localhost:8000  |
| Frontend | http://localhost:8501  |
| API Docs | http://localhost:8000/docs |

---

## Switching to Pinecone

```bash
# In .env
VECTOR_DB_MODE=pinecone
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=neuro-rag

# Rebuild index
python scripts/build_index.py
```

---

## Project Structure

```
neuro-rag/
├── scraper/
│   ├── base.py              # Article dataclass + BaseScraper
│   ├── pubmed.py            # NCBI E-utilities
│   ├── arxiv.py             # ArXiv Atom API
│   ├── medrxiv.py           # MedRxiv REST API
│   ├── semantic_scholar.py  # S2 Academic Graph API
│   └── __init__.py          # Pipeline runner + dedup
├── embeddings/
│   └── engine.py            # SentenceTransformer wrapper
├── vectordb/
│   ├── faiss_store.py       # FAISS IndexFlatIP
│   ├── pinecone_store.py    # Pinecone serverless
│   └── __init__.py          # Factory function
├── api/
│   └── main.py              # FastAPI app
├── frontend/
│   └── app.py               # Streamlit UI
├── scripts/
│   └── build_index.py       # CLI pipeline
├── tests/
│   └── test_pipeline.py     # Unit + integration tests
├── .github/workflows/ci.yml # GitHub Actions CI/CD
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── config.py
├── requirements.txt
└── .env.example
```

---

## API Reference

| Method | Endpoint  | Description                          |
|--------|-----------|--------------------------------------|
| GET    | `/`       | Health check                         |
| GET    | `/search?q=...&top_k=5` | Semantic search   |
| GET    | `/stats`  | Index size and mode                  |
| GET    | `/docs`   | Auto-generated Swagger UI            |

---

## Technologies

| Layer      | Technology                              |
|------------|-----------------------------------------|
| Scraping   | `requests`, `BeautifulSoup`, NCBI API, ArXiv API |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector DB  | `FAISS` (local) / `Pinecone` (cloud)   |
| Backend    | `FastAPI` + `uvicorn`                   |
| Frontend   | `Streamlit`                             |
| CI/CD      | GitHub Actions + Docker                 |

---

## GitHub Actions Secrets

Add these in **Settings → Secrets → Actions** for Docker Hub push:

| Secret              | Value                    |
|---------------------|--------------------------|
| `DOCKERHUB_USERNAME`| Your Docker Hub username |
| `DOCKERHUB_TOKEN`   | Docker Hub access token  |

---

## Pushing to GitHub

```bash
git init
git add .
git commit -m "feat: initial NeuroRAG implementation"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/neuro-rag.git
git push -u origin main
```
