import os
import time
import pickle
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, Docx2txtLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# -----------------------------
# CONFIGURATION
# -----------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"
DEFAULT_DOC_FOLDER = "docs"

# -----------------------------
# Load documents + FAISS
# -----------------------------

def create_prompt(context, query):
    return f"""You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:"""
    
@st.cache_resource
def load_and_index_documents(folder_path, reload=False):

    start = time.time()
    safe_folder = folder_path.replace("/", "_").replace(" ", "_")
    vector_cache = f"faiss_index__{safe_folder}.pkl"

    # If cache exists, load it
    if not reload and Path(vector_cache).exists():
        cache_age = time.time() - os.path.getmtime(vector_cache)
        if cache_age > 7 * 86400:
            st.warning("Your FAISS index cache is over a week old. Consider refreshing it.")

        with open(vector_cache, "rb") as f:
            vector_store = pickle.load(f)
        duration = time.time() - start
        return vector_store, duration, f"Loaded from cache ({Path(vector_cache).name})", [], [], 0

    # Otherwise: rebuild
    docs = []
    all_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            all_files.append(os.path.join(root, file))
    total_files = len(all_files)
    progress_bar = st.progress(0)
    added_count = 0
    skipped_files = []

    for i, file in enumerate(all_files):
        path = file
        ext = Path(file).suffix.lower()

        try:
            if ext == ".txt":
                loader = TextLoader(path)
            elif ext == ".pdf":
                loader = PyMuPDFLoader(path)
            elif ext == ".docx":
                loader = Docx2txtLoader(path)
            else:
                skipped_files.append((file, "Unsupported file type"))
                continue

            file_docs = loader.load()
            if file_docs:
                docs.extend(file_docs)
                added_count += 1
            else:
                skipped_files.append((file, "No content extracted"))

        except Exception as e:
            skipped_files.append((file, f"Error: {str(e)}"))

        progress_bar.progress((i + 1) / total_files)

    progress_bar.empty()

    if not docs:
        st.error("🚫 No valid documents were loaded. Aborting.")
        st.stop()

    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vector_store = FAISS.from_documents(docs, embedding_model)

    with open(vector_cache, "wb") as f:
        pickle.dump(vector_store, f)

    duration = time.time() - start
    return vector_store, duration, f"Rebuilt from documents → {Path(vector_cache).name}", skipped_files, total_files, added_count


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="LLM Doc QA", layout="wide")
st.title("📄 Ask Your Documents (Fast, Local RAG)")

folder_input = st.text_input("🔍 Folder with documents", value=DEFAULT_DOC_FOLDER)
col1, col2, _, col3 = st.columns([2, 2, 4, 1]) 
with col1:
    submitted = st.button("📂 Load Documents")
with col2:
    reload_submitted = st.button("🔄 Reload Documents (Force)")
with col3:
    clear_triggered = st.button("🧹 Clear Input")
        
if submitted or reload_submitted:
    reload_flag = reload_submitted
    if not os.path.isdir(folder_input):
        st.error("⚠️ Invalid folder path.")
        st.stop()
    else:
        result = load_and_index_documents(folder_input, reload=reload_flag)
        vector_store, indexing_time, indexing_source, skipped, total, added = result
        st.session_state["vector_store"] = vector_store
        
        st.success(f"✅ Vector index ready ({indexing_source}, took {indexing_time:.2f} sec)")
        if "Loaded" not in indexing_source:
            st.info(f"📂 Processed {total} files. ✅ {added} added, ❌ {len(skipped)} skipped.")
        
        if skipped:
            with st.expander("⚠️ Skipped Files", expanded=False):
                for fname, reason in skipped:
                    st.write(f"- {fname}: {reason}")


if clear_triggered:
    st.experimental_rerun()

query = st.text_input("Ask a question based on your documents:")

if query:
    if "vector_store" not in st.session_state:
        st.warning("📂 Please load a document folder first.")
        st.stop()

    vector_store = st.session_state["vector_store"]


# LLM Setup
client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN)

if query:
    with st.spinner("🔎 Searching and generating..."):
        docs = vector_store.similarity_search_with_score(query, k=5)
        chunks = [doc.page_content for doc, _ in docs]
        scores = [score for _, score in docs]

        context = "\n\n".join(chunks)

        prompt = create_prompt(context, query) 

        if len(prompt.split()) + 256 > 32000:
            st.error("⚠️ Prompt too long for the model. Try a shorter question or fewer chunks.")
            st.stop()

        response = client.text_generation(
            prompt,
            max_new_tokens=256,
            temperature=0.3,
            repetition_penalty=1.1
        )

        st.markdown("### 📢 Answer:")
        st.write(response.strip())

        with st.expander("Information About Search"):
            st.markdown("**Token Estimate**")
            token_estimate = len(prompt.split())
            st.write(f"{token_estimate} tokens estimated")
        
            st.markdown("**Top Retrieved Chunks**")
            for i, ((doc, score)) in enumerate(docs):
                st.markdown(f"**Chunk {i+1} — Similarity Score: {score:.4f}**")
                st.write(doc.page_content[:1000])
