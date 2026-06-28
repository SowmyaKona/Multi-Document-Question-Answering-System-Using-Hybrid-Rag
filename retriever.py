from langchain_community.retrievers import BM25Retriever

def get_dense_retriever(vector_store,search_type="similarity",k=5):
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )

    return retriever

def get_sparse_retriever(chunks,k= 10):
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k
    return bm25_retriever