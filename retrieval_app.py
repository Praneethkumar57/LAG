# import streamlit as st
# from langchain_ollama import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS

# # --- Configuration ---
# DB_PATH = "/home/aizen/Documents/RAG/RAG_QueryRewrite/Paper_finder/papers_vector_db"
# EMBED_MODEL = "nomic-embed-text-v2-moe"

# st.set_page_config(page_title="Paper Search Engine", layout="centered")
# st.title("Research Paper Discovery")

# @st.cache_resource
# def load_db():
#     embeddings = OllamaEmbeddings(model=EMBED_MODEL)
#     return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

# # Try to load the DB
# try:
#     db = load_db()
    
#     query = st.text_input("What research topic are you looking for?", placeholder="e.g. Distributed systems and fault tolerance")

#     if query:
#         # Search for top 5 matches
#         results = db.similarity_search(query, k=5)
        
#         st.subheader(f"Top 5 Relevant Papers:")
        
#         for i, doc in enumerate(results):
#             with st.container(border=True):
#                 col1, col2 = st.columns([1, 4])
#                 col1.metric("Rank", i + 1)
#                 col2.markdown(f"### {doc.metadata['title']}")
#                 with col2.expander("Show Indexed Keywords"):
#                     st.write(doc.metadata['keywords'])

# except Exception as e:
#     st.error("Database not found! Please run 'python3 database_builder.py' first.")

import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
DB_PATH = "/home/aizen/Documents/RAG/RAG_QueryRewrite/Paper_finder/paper_vector_db_lmda"
EMBED_MODEL = "nomic-embed-text-v2-moe"

st.set_page_config(page_title="LMDA Paper Explorer", layout="wide")
st.title("LMDA-Enriched Research Discovery")

@st.cache_resource
def load_db():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    # Ensure this path matches where your database_builder.py saved the index
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

try:
    db = load_db()
    
    query = st.text_input("Describe the research concept or methodology you need:", 
                          placeholder="e.g. Transformers for time-series forecasting")

    if query:
        # CRITICAL: Prepend the search_query prefix for Nomic MOE
        # This aligns the query vector with the 'search_document' vectors in your DB
        prefixed_query = f"search_query: {query}"
        
        with st.spinner("Analyzing dimensions and searching..."):
            results = db.similarity_search(prefixed_query, k=5)
        
        st.subheader("Top 5 Relevant Papers")
        
        for i, doc in enumerate(results):
            with st.container(border=True):
                col1, col2 = st.columns([1, 5])
                col1.metric("Relevance", f"{5-i}/5")
                
                col2.markdown(f"### {doc.metadata.get('title', 'Unknown Title')}")
                
                # Highlight that this was analyzed via LMDA
                if doc.metadata.get('analysis_type') == 'LMDA':
                    st.caption("Indexed via Multi-Dimensional Factor Analysis")
                
                with col2.expander("Technical Profile (Tech/Metrics/Concepts)"):
                    st.write(doc.metadata.get('technical_profile', "No profile available."))

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Check if the DB_PATH is correct and the database has been built.")
