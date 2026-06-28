from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder(
    "BAAI/bge-reranker-base"
)

def rerank_documents(query,docs,top_k=5):
    
    pairs = [(query, doc.page_content)for doc in docs]

    scores = reranker_model.predict(pairs)

    scored_docs = list(zip(docs, scores))

    scored_docs.sort(key=lambda x: x[1],reverse=True)

    return scored_docs[:top_k]
    
    