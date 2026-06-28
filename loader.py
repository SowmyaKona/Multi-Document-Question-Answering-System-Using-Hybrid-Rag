from langchain_community.document_loaders import PyPDFLoader

def load_multiple_pdfs(pdf_paths):
    documents = []

    for pdf_path in pdf_paths:
        print(f"\nLoading: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(f"Documents extracted: {len(docs)}")
        documents.extend(docs)
    print(f"\nTotal documents loaded: {len(documents)}")
    return documents