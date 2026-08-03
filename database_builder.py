
import os
import re
import pickle
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- Configuration ---
PAPERS_DIR = "./research_papers"
DB_PATH = "./paper_vector_db_lmda"
LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text-v2-moe"

llm = OllamaLLM(model=LLM_MODEL)
embeddings = OllamaEmbeddings(model=EMBED_MODEL)


def extract_abstract(text):
    """
    Heuristic: grab the text right after the literal word 'Abstract'.
    Falls back to the first ~2000 chars if no 'Abstract' header is found.
    """
    match = re.search(r"\babstract\b", text, re.IGNORECASE)
    if match:
        start = match.end()
        return text[start:start + 2000].strip()
    return text[:2000].strip()


def extract_conclusion(text):
    """
    Heuristic: look for a 'Conclusion' section header and grab the text
    that follows, trimmed before References if present. Falls back to
    the last chunk of text before a likely References section.
    """
    match = re.search(r"\b(\d\s*)?conclusion(s)?\b", text, re.IGNORECASE)
    if match:
        start = match.end()
        snippet = text[start:start + 2000]
        ref_match = re.search(r"\breferences\b", snippet, re.IGNORECASE)
        if ref_match:
            snippet = snippet[:ref_match.start()]
        return snippet.strip()

    ref_match = re.search(r"\breferences\b", text, re.IGNORECASE)
    end = ref_match.start() if ref_match else len(text)
    return text[max(0, end - 2000):end].strip()


def generate_natural_summary(text):
    prompt = PromptTemplate.from_template(
        "Summarize the following research paper in 3-5 plain-English sentences. "
        "Cover: (1) what problem the paper addresses, (2) what method or approach "
        "it uses, and (3) what the main finding or result is. "
        "Write it as flowing prose, NOT a list, NOT bullet points, and avoid "
        "restating the paper's title verbatim.\n\n"
        "Text: {text}"
    )
    chain = prompt | llm | StrOutputParser()
    # First ~4000 chars usually covers abstract + intro + enough of the
    # method section for the LLM to describe problem/method/finding.
    return chain.invoke({"text": text[:4000]}).strip()


def build_database():
    all_docs = []
    if not os.path.exists(PAPERS_DIR):
        print(f"Error: Folder {PAPERS_DIR} not found.")
        return

    pdf_files = [f for f in os.listdir(PAPERS_DIR) if f.endswith('.pdf')]

    for filename in pdf_files:
        print(f"\n{'=' * 70}")
        print(f"Processing: {filename}")
        print(f"{'=' * 70}")
        try:
            loader = PyPDFLoader(os.path.join(PAPERS_DIR, filename))
            pages = loader.load()
            full_text = " ".join([p.page_content for p in pages])

            abstract = extract_abstract(full_text)
            conclusion = extract_conclusion(full_text)
            summary = generate_natural_summary(full_text)

            # --- Manual QA printout ---
            # Print what got extracted so you can eyeball whether the
            # heuristics grabbed the right section, before trusting it
            # in the index. Cheap sanity check, catches bad extractions
            # early (e.g. abstract heuristic grabbing author list instead).

            
            # print("\n--- ABSTRACT (first 300 chars) ---")
            # print(abstract[:300] + ("..." if len(abstract) > 300 else ""))

            # print("\n--- CONCLUSION (first 300 chars) ---")
            # print(conclusion[:300] + ("..." if len(conclusion) > 300 else ""))

            # print("\n--- NATURAL-LANGUAGE SUMMARY (full) ---")
            # print(summary)

            base_metadata = {
                "title": filename,
                "abstract": abstract,
                "conclusion": conclusion,
                "summary": summary,
                "analysis_type": "Natural_Language_Summary",
            }

            # --- Create 3 chunks per paper: abstract, summary, conclusion ---
            all_docs.append(Document(
                page_content=f"Title: {filename}. Abstract: {abstract}",
                metadata={**base_metadata, "chunk_type": "abstract"},
            ))
            all_docs.append(Document(
                page_content=f"Title: {filename}. Summary: {summary}",
                metadata={**base_metadata, "chunk_type": "summary"},
            ))
            all_docs.append(Document(
                page_content=f"Title: {filename}. Conclusion: {conclusion}",
                metadata={**base_metadata, "chunk_type": "conclusion"},
            ))

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    print(f"\n{'=' * 70}")
    print("Building Vector Database...")
    print(f"{'=' * 70}")
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    vectorstore.save_local(DB_PATH)

    with open(os.path.join(DB_PATH, "bm25_corpus.pkl"), "wb") as f:
        pickle.dump(all_docs, f)

    print(f"✅ Database saved to {DB_PATH} ({len(all_docs)} chunks from {len(pdf_files)} papers)")


if __name__ == "__main__":
    build_database()
