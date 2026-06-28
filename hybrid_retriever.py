def reciprocal_rank_fusion(dense_docs,sparse_docs,k=60):

    scores = {}

    for rank, doc in enumerate(dense_docs):
        doc_id = doc.page_content
        scores[doc_id] = (scores.get(doc_id, 0)+ 1 / (k + rank))

    for rank, doc in enumerate(sparse_docs):
        doc_id = doc.page_content
        scores[doc_id] = (scores.get(doc_id, 0)+ 1 / (k + rank))
    
    unique_docs = {}

    for doc in dense_docs + sparse_docs:
        unique_docs[doc.page_content] = doc

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [unique_docs[text]for text, _ in ranked]


