# # 2.now documents --> splitter --> chunks

# from langchain_text_splitters import SentenceTransformersTokenTextSplitter

# def split_documents(
#     documents,
#     chunk_size=512,
#     chunk_overlap=100
# ):
#     splitter = SentenceTransformersTokenTextSplitter(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         tokens_per_chunk=chunk_size,
#         chunk_overlap=chunk_overlap
#     )
#     return splitter.split_documents(documents)


from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(
    documents,
    chunk_size=512,
    chunk_overlap=100
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    return chunks