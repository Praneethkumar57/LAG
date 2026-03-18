# import os
# from langchain_ollama import OllamaLLM, OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.documents import Document

# # --- Configuration ---
# PAPERS_DIR = "./research_papers"  # Put your 20 PDFs here
# DB_PATH = "paper_vector_db"
# LLM_MODEL = "llama3.2"
# EMBED_MODEL = "nomic-embed-text-v2-moe"

# llm = OllamaLLM(model=LLM_MODEL)
# embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# def extract_keywords(text):
#     prompt = PromptTemplate.from_template(
#         "Extract exactly 30 specific technical keywords or keyphrases from this research paper text. "
#         "Focus on methodology, results, and core concepts. Return only the keywords separated by commas.\n\n"
#         "Text: {text}"
#     )
#     chain = prompt | llm | StrOutputParser()
#     # We use the first 3000 characters to get the abstract/intro for keyword extraction
#     return chain.invoke({"text": text[:3000]})

# def build_database():
#     all_docs = []
    
#     if not os.path.exists(PAPERS_DIR):
#         print(f"Error: Folder {PAPERS_DIR} not found.")
#         return

#     pdf_files = [f for f in os.listdir(PAPERS_DIR) if f.endswith('.pdf')]
    
#     for filename in pdf_files:
#         print(f"Processing: {filename}...")
#         loader = PyPDFLoader(os.path.join(PAPERS_DIR, filename))
#         pages = loader.load()
#         full_text = " ".join([p.page_content for p in pages])
        
#         # 1. Extract 30 keywords
#         keywords = extract_keywords(full_text)
        
#         # 2. Create a document where the searchable content is the keywords
#         # and the metadata is the paper title
#         doc = Document(
#             page_content=f"Title: {filename}. Keywords: {keywords}",
#             metadata={"title": filename, "keywords": keywords}
#         )
#         all_docs.append(doc)

#     # 3. Create and Save FAISS Database
#     print("Creating Vector Database...")
#     vectorstore = FAISS.from_documents(all_docs, embeddings)
#     vectorstore.save_local(DB_PATH)
#     print(f"Database saved to {DB_PATH}")

# if __name__ == "__main__":
#     build_database()


import os
import spacy
from collections import Counter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- Configuration ---
PAPERS_DIR = "./research_papers"
DB_PATH = "paper_vector_db_lmda"
LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text-v2-moe"

nlp = spacy.load("en_core_web_sm")
llm = OllamaLLM(model=LLM_MODEL)
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

def lmda_factor_analysis(text):
    # Analyzing a larger chunk to ensure we catch the 'Evaluation' section where metrics live
    doc = nlp(text[:20000]) 
    
    # We focus on PROPN (Proper Nouns for Tech) and NOUN (for Concepts/Metrics)
    lemmas = [token.lemma_.lower() for token in doc 
              if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and len(token.text) > 1]
    
    counts = Counter(lemmas)
    significant_terms = [word for word, count in counts.most_common(120)]
    return ", ".join(significant_terms)

def extract_tech_metrics_with_lmda(text):
    lmda_features = lmda_factor_analysis(text)
    
    # Updated Prompt to target Technologies, Metrics, and Domains
    prompt = PromptTemplate.from_template(
        "You are a technical research assistant. Analyze the provided lexical features "
        "and extract exactly 30 technical keywords categorized as follows:\n"
        "1. TECHNOLOGIES: Frameworks, libraries, models, or hardware used.\n"
        "2. METRICS: Quantitative measures (e.g., F1-score, Latency, Accuracy, BLEU).\n"
        "3. MAIN CONCEPTS: The primary research fields or problems addressed.\n\n"
        "Provide a clean, categorized list.\n\n"
        "Lexical Features: {features}"
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"features": lmda_features})

def build_database():
    all_docs = []
    if not os.path.exists(PAPERS_DIR):
        print(f"Error: Folder {PAPERS_DIR} not found.")
        return

    pdf_files = [f for f in os.listdir(PAPERS_DIR) if f.endswith('.pdf')]
    
    for filename in pdf_files:
        print(f"Extracting Tech & Metrics: {filename}...")
        try:
            loader = PyPDFLoader(os.path.join(PAPERS_DIR, filename))
            pages = loader.load()
            full_text = " ".join([p.page_content for p in pages])
            
            # Extract the specific technical attributes
            attributes = extract_tech_metrics_with_lmda(full_text)
            
            doc = Document(
                page_content=f"Title: {filename}. Technical Profile: {attributes}",
                metadata={
                    "title": filename, 
                    "technical_profile": attributes,
                    "analysis_type": "LMDA_Tech_Metrics"
                }
            )
            all_docs.append(doc)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    print("Building Vector Database...")
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    vectorstore.save_local(DB_PATH)
    print(f"✅ Database saved to {DB_PATH}")

if __name__ == "__main__":
    build_database()
