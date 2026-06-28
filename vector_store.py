from langchain_chroma import Chroma
import shutil
import os 
CHROMA_DB_PATH = "./chroma_db"

def create_vector_store(
    chunks,
    embedding_model,
    collection_name="rag_documents"
):
    # Delete previous database if it exists
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH, ignore_errors=True)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_PATH,
        collection_name=collection_name
    )

    return vector_store