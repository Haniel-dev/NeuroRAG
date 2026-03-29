"""Streamlit frontend for the Neurodivergent RAG system."""
import os
import time

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page config ──────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="NeuroRAG — Neurodivergent Research Search",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────── #
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3 { font-family: 'DM Serif Display', serif; }

  .stApp { background: #0f1117; color: #e8e8f0; }

  .hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    font-style: italic;
    background: linear-gradient(135deg, #7c6ef7, #c084fc, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
  }
  .hero-subtitle { color: #8b8fa8; font-size: 1.05rem; margin-bottom: 2rem; }

  .result-card {
    background: #1a1d2e;
    border: 1px solid #2d3150;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
  }
  .result-card:hover { border-color: #7c6ef7; }

  .result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #e2e2f0;
    margin-bottom: 0.4rem;
  }
  .result-meta { font-size: 0.8rem; color: #6b6f8a; margin-bottom: 0.5rem; }
  .result-abstract { font-size: 0.9rem; color: #9da0b8; line-height: 1.6; }

  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-right: 6px;
  }
  .badge-pubmed           { background:#1a3a2a; color:#4ade80; }
  .badge-arxiv            { background:#3a1a1a; color:#f87171; }
  .badge-medrxiv          { background:#1a2a3a; color:#60a5fa; }
  .badge-semantic_scholar { background:#2a1a3a; color:#c084fc; }

  .score-chip {
    float: right;
    background: #7c6ef7;
    color: white;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
  }
  div[data-testid="stTextInput"] input {
    background: #1a1d2e;
    border: 1px solid #2d3150;
    border-radius: 8px;
    color: #e2e2f0;
    font-size: 1rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────── #
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    top_k = st.slider("Results to return", 1, 20, 5)
    st.markdown("---")

    st.markdown("### 📊 Index Stats")
    try:
        stats = requests.get(f"{API_BASE}/stats", timeout=4).json()
        st.metric("Mode",    stats.get("mode", "—").upper())
        st.metric("Vectors", f"{stats.get('total_vectors', 0):,}")
    except Exception:
        st.warning("API offline — start the FastAPI server first.")

    st.markdown("---")
    st.caption("Built with sentence-transformers, FAISS/Pinecone, FastAPI & Streamlit.")


# ── Hero ─────────────────────────────────────────────────────────────── #
st.markdown('<p class="hero-title">NeuroRAG</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Semantic search over neurodivergent scientific literature '
    '— PubMed · ArXiv · MedRxiv · Semantic Scholar</p>',
    unsafe_allow_html=True,
)

# ── Search bar ───────────────────────────────────────────────────────── #
query = st.text_input(
    "",
    placeholder="e.g. executive function deficits in ADHD adults  …",
    key="search_input",
)

col1, col2, col3 = st.columns([1, 1, 4])
search_btn = col1.button("🔍 Search", use_container_width=True, type="primary")

# Sample queries
SAMPLES = [
    "working memory autism",
    "sensory processing ADHD",
    "dyslexia reading intervention",
    "executive function neurodivergent",
]
with col2:
    chosen = st.selectbox("Try a sample", ["—"] + SAMPLES, label_visibility="collapsed")
    if chosen != "—":
        query = chosen

# ── Results ──────────────────────────────────────────────────────────── #
if search_btn or (query and query != st.session_state.get("last_query")):
    st.session_state["last_query"] = query
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Searching the literature …"):
            t0 = time.time()
            try:
                resp = requests.get(
                    f"{API_BASE}/search",
                    params={"q": query, "top_k": top_k},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach the API. Make sure the FastAPI server is running on port 8000.")
                st.stop()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        elapsed = time.time() - t0
        results = data.get("results", [])
        total   = data.get("total", 0)

        st.markdown(
            f"**{total} results** for *{query}* &nbsp;·&nbsp; "
            f"<span style='color:#6b6f8a;font-size:0.85rem;'>{elapsed:.2f}s</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        if not results:
            st.info("No results found — try different search terms or rebuild the index.")
        else:
            for r in results:
                source  = r.get("source", "unknown")
                badge   = f'<span class="badge badge-{source}">{source.replace("_", " ")}</span>'
                score   = r.get("score", 0)
                authors = ", ".join(r.get("authors", [])[:3])
                if len(r.get("authors", [])) > 3:
                    authors += " et al."

                abstract = r.get("abstract", "")
                abstract_preview = abstract[:300] + "…" if len(abstract) > 300 else abstract

                doi_link = (
                    f'<a href="https://doi.org/{r["doi"]}" target="_blank" '
                    f'style="color:#7c6ef7;text-decoration:none;">DOI ↗</a>'
                    if r.get("doi") else
                    f'<a href="{r.get("url","#")}" target="_blank" '
                    f'style="color:#7c6ef7;text-decoration:none;">View ↗</a>'
                )

                st.markdown(f"""
<div class="result-card">
  <div class="score-chip">{score:.3f}</div>
  {badge}
  <div class="result-title">{r.get("title","")}</div>
  <div class="result-meta">{authors} &nbsp;·&nbsp; {r.get("published_date","")[:7]} &nbsp;·&nbsp; {doi_link}</div>
  <div class="result-abstract">{abstract_preview}</div>
</div>
""", unsafe_allow_html=True)
