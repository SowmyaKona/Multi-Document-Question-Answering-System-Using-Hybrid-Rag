# 📚 Multi-Document Question Answering System using Hybrid RAG

## 📌 Overview

This project is a **Hybrid Retrieval-Augmented Generation (Hybrid RAG)** based Question Answering System that enables users to upload multiple PDF documents and ask questions in natural language.

Unlike a traditional Large Language Model (LLM), which relies only on its pre-trained knowledge, this application retrieves the most relevant information from the uploaded documents before generating an answer. This improves response accuracy and reduces hallucinations.

---

## 🚀 Features

* Upload and process multiple PDF documents
* Automatic document loading and chunking
* Embedding generation and vector indexing
* ChromaDB vector database integration
* Hybrid Retrieval using:

  * Dense Retrieval
  * BM25 Sparse Retrieval
* Reciprocal Rank Fusion (RRF)
* Cross-Encoder Re-ranking
* Gemini API for context-aware answer generation
* Retrieval tuning:

  * Chunk Size
  * Chunk Overlap
  * Top-K Retrieval
  * Search Type (Similarity / MMR)
* Retrieval Analysis Dashboard
* Source document visualization

---

## 🏗️ Project Architecture

```text
User Uploads PDFs
        │
        ▼
PyPDFLoader
        │
        ▼
Text Chunking
        │
        ▼
Embedding Generation
        │
        ▼
ChromaDB
        │
 ┌──────┴──────┐
 ▼             ▼
Dense      BM25 Retrieval
Retrieval
 └──────┬──────┘
        ▼
Reciprocal Rank Fusion (RRF)
        │
        ▼
Cross-Encoder Re-ranking
        │
        ▼
Gemini API
        │
        ▼
Generated Answer
```

---

## ⚙️ Tech Stack

* Python
* LangChain
* Gemini API
* ChromaDB
* Embedding Model
* PyPDFLoader
* RecursiveCharacterTextSplitter
* BM25
* Reciprocal Rank Fusion (RRF)
* Cross-Encoder Re-ranking
* Streamlit

---

## 📂 Project Structure

```text
├── app.py
├── loader.py
├── splitter.py
├── embeddings.py
├── vector_store.py
├── retriever.py
├── hybrid_retriever.py
├── reranker.py
├── rag_chain.py
├── llm.py
├── requirements.txt
├── assets/
├── data/
└── chroma_db/
```

---

## 🔄 Workflow

1. Upload one or more PDF documents.
2. Documents are loaded using **PyPDFLoader**.
3. Documents are split into smaller chunks.
4. Each chunk is converted into embeddings.
5. Embeddings are stored in **ChromaDB**.
6. User submits a question.
7. The query is converted into an embedding.
8. Hybrid Retrieval is performed using:

   * Dense Retrieval
   * BM25 Sparse Retrieval
9. Results are combined using **Reciprocal Rank Fusion (RRF)**.
10. Retrieved chunks are re-ranked using a **Cross-Encoder**.
11. The highest-ranked chunks are provided to the **Gemini API**.
12. Gemini generates a context-aware answer.

---

## 🎯 Retrieval Tuning

The application supports configurable retrieval parameters:

* Chunk Size
* Chunk Overlap
* Top-K Retrieval
* Search Type (Similarity / MMR)

These parameters help optimize retrieval quality for different document collections.

---

## 📊 Retrieval Analysis

The application provides a Retrieval Analysis dashboard displaying:

* Number of Dense Retrieval results
* Number of BM25 Retrieval results
* Number of RRF fused results
* Final re-ranked chunks
* Retrieved document chunks
* Source document references

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/multi-document-hybrid-rag-qa-system.git
```

Navigate to the project folder:

```bash
cd multi-document-hybrid-rag-qa-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

Run the application:

```bash
streamlit run app.py
```

---

## 📸 Application

The application allows users to:

* Upload multiple PDF documents
* Ask questions in natural language
* View retrieved chunks
* Analyze retrieval results
* View retrieved source documents

---

## 🚀 Future Improvements

* RAGAS Evaluation Integration
* Metadata Filtering
* Agentic RAG using LangGraph



**Sowmya Kona**
