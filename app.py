import os
import shutil
import streamlit as st

from loader import load_multiple_pdfs
from splitter import split_documents
from embeddings import get_embedding_model
from vector_store import create_vector_store
from retriever import (
    get_dense_retriever,
    get_sparse_retriever
)
from hybrid_retriever import reciprocal_rank_fusion
from reranker import rerank_documents
from llm import get_llm
from rag_chain import create_rag_chain

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Multi-Document Hybrid RAG Question Answering System",
    layout="wide"
)

st.title("📚 Multi-Document Hybrid RAG Question Answering System")

# -------------------------
# SIDEBAR
# -------------------------

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

process_button = st.sidebar.button("Process Documents")
    
# -----------------------------
# Retrieval Settings
# -----------------------------

chunk_size = st.sidebar.slider(
    "Chunk Size",
    min_value=200,
    max_value=1000,
    value=512,
    step=100
)

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    min_value=0,
    max_value=200,
    value=100,
    step=20
)

top_k = st.sidebar.slider(
    "Top-K Retrieval",
    min_value=1,
    max_value=20,
    value=5
)

search_type = st.sidebar.selectbox(
    "Search Type",
    [
        "similarity",
        "mmr"
    ]
)

# -------------------------
# PROCESS DOCUMENTS
# -------------------------

if uploaded_files and process_button:
    with st.spinner("Processing Documents..."):

        # Clear old files
        if os.path.exists("data"):
            shutil.rmtree("data")

        os.makedirs("data", exist_ok=True)

        pdf_paths = []

        for file in uploaded_files:
            file_path = os.path.join("data",file.name)

            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            pdf_paths.append(file_path)

        # Load PDFs
        documents = load_multiple_pdfs(pdf_paths)

        # Split Documents
        chunks = split_documents(documents,chunk_size=chunk_size,chunk_overlap=chunk_overlap)

        # Embeddings
        embedding_model = (get_embedding_model())     

        # Vector Store
        vector_store = (create_vector_store(chunks,embedding_model))

        # Dense Retriever
        dense_retriever = get_dense_retriever(
            vector_store,
            search_type=search_type,
            k=top_k
        )

        # Sparse Retriever
        sparse_retriever = get_sparse_retriever(chunks, k=top_k)

        # LLM
        llm = get_llm()

        # RAG Chain
        rag_chain = create_rag_chain(llm)

        # Save session state
        st.session_state.processed = True

        st.session_state.doc_count = (len(documents))

        st.session_state.chunk_count = (len(chunks))

        st.session_state.dense_retriever = (dense_retriever)

        st.session_state.sparse_retriever = (sparse_retriever)

        st.session_state.rag_chain = (rag_chain)

    st.success("Documents processed successfully!")

# -------------------------
# SHOW STATS
# -------------------------

if st.session_state.get("processed",False):
    st.write(f"Documents Loaded: {st.session_state.doc_count}")
    st.write(f"Chunks Created: {st.session_state.chunk_count}")

# -------------------------
# QUESTION
# -------------------------

if st.session_state.get("processed",False):

    question = st.text_input("Ask a question about your documents")

    if question:
        dense_retriever = (st.session_state.dense_retriever)

        sparse_retriever = (st.session_state.sparse_retriever)

        rag_chain = (st.session_state.rag_chain)

        # Dense Retrieval
        dense_docs = (dense_retriever.invoke(question))

        # Sparse Retrieval
        sparse_docs = (sparse_retriever.invoke(question))

        # RRF Fusion
        fused_docs = (reciprocal_rank_fusion(dense_docs,sparse_docs))

        # Re-ranking
        reranked_docs = (rerank_documents(question,fused_docs,top_k=5))
        
        # Build Context
        context = "\n\n".join(
            doc.page_content
            for doc, score
            in reranked_docs
        )

        # Generate Answer

        answer = rag_chain.invoke(
            {
                "context": context,
                "question": question
            }
        )

        contexts = [doc.page_content for doc, score in reranked_docs]

        # -------------------------
        # TABS
        # -------------------------

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "💬 Answer",
                "📊 Retrieval Analysis",
                "📄 Sources",
                "📈 RAGAS"
            ]
        )

        # -------------------------
        # ANSWER TAB
        # -------------------------
        with tab1:
            st.subheader("Generated Answer")
            st.write(answer)

        # -------------------------
        # ANALYSIS TAB
        # -------------------------

        with tab2:

            col1, col2, col3, col4 = (st.columns(4))

            col1.metric("Dense Chunks",len(dense_docs))

            col2.metric("Sparse Chunks",len(sparse_docs))

            col3.metric("RRF Chunks",len(fused_docs))

            col4.metric("Re-Ranked Top k",len(reranked_docs))

            st.divider()
            st.subheader("Re-ranked Chunks")

            for idx, (doc,score) in enumerate(reranked_docs,start=1):
                with st.expander(f"Chunk {idx}"):
                    st.write(f"Score: {score:.4f}")
                    st.write(doc.page_content)

        # -------------------------
        # SOURCES TAB
        # -------------------------
        with tab3:
            for idx, (doc,score) in enumerate(reranked_docs,start=1):

                source = (doc.metadata.get("source","Unknown"))
                page = (doc.metadata.get("page","Unknown"))
                with st.expander(f"Source {idx}"):
                    st.write(f"Source: {source}")
                    st.write(f"Page: {page}")
        
        # ---------------
        # RAGAS Tab
        # ---------------

        with tab4:

            st.subheader("RAGAS Evaluation")

            st.info(
                "Evaluation metrics (Faithfulness, Answer Relevancy, "
                "Context Precision, Context Recall) will be added soon."
            )


            