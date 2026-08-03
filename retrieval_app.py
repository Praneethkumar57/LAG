
import streamlit as st
import pickle
import os
from rank_bm25 import BM25Okapi
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
DB_PATH = "./paper_vector_db_lmda" 
EMBED_MODEL = "nomic-embed-text-v2-moe"

st.set_page_config(page_title="LMDA Paper Explorer", layout="wide")
st.title("LMDA-Enriched Research Discovery")

@st.cache_resource
def load_db():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

@st.cache_resource
def load_bm25():
    with open(os.path.join(DB_PATH, "bm25_corpus.pkl"), "rb") as f:
        docs = pickle.load(f)
    tokenized = [doc.page_content.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25, docs

def hybrid_search(query, db, bm25, docs, k=5, k_rrf=60):
    """
    Combine FAISS (semantic) and BM25 (lexical) rankings using
    Reciprocal Rank Fusion (RRF) -- simple, no score-scale tuning needed.
    """
    # Semantic side
    prefixed_query = f"search_query: {query}"
    semantic_results = db.similarity_search(prefixed_query, k=20)

    # Lexical side
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]

    # RRF fusion: score = sum(1 / (k_rrf + rank)) across both lists
    rrf_scores = {}

    for rank, doc in enumerate(semantic_results):
        title = doc.metadata.get("title")
        rrf_scores[title] = rrf_scores.get(title, 0) + 1.0 / (k_rrf + rank)

    for rank, idx in enumerate(bm25_ranked_idx):
        title = docs[idx].metadata.get("title")
        rrf_scores[title] = rrf_scores.get(title, 0) + 1.0 / (k_rrf + rank)

    # Map back to full doc objects (prefer the FAISS version if present, else BM25's)
    title_to_doc = {d.metadata.get("title"): d for d in docs}
    for d in semantic_results:
        title_to_doc[d.metadata.get("title")] = d

    ranked_titles = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [title_to_doc[title] for title, _ in ranked_titles]

try:
    db = load_db()
    bm25, bm25_docs = load_bm25()

    query = st.text_input(
        "Describe the research concept or methodology you need:",
        placeholder="e.g. Transformers for time-series forecasting"
    )

    if query:
        with st.spinner("Running hybrid (semantic + lexical) search..."):
            results = hybrid_search(query, db, bm25, bm25_docs, k=5)

        st.subheader("Top 5 Relevant Papers (Hybrid Search)")

        for i, doc in enumerate(results):
            with st.container(border=True):
                col1, col2 = st.columns([1, 5])
                col1.metric("Rank", i + 1)
                col2.markdown(f"### {doc.metadata.get('title', 'Unknown Title')}")
                with col2.expander("Technical Profile (Tech/Metrics/Concepts)"):
                    st.write(doc.metadata.get('technical_profile', "No profile available."))

except FileNotFoundError:
    st.error("Database not found! Please run `python3 database_builder.py` first.")
except Exception as e:
    st.error(f"Error: {e}")
