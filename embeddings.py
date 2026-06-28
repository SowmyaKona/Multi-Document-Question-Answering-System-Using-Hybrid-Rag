# 3. chunks --> embedding model --> vector

from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(   # It only loads the model.
        model_name = "BAAI/bge-small-en-v1.5",
        encode_kwargs = {"normalize_embeddings":True}
    )

    return embeddings