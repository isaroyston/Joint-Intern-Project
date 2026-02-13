import os
from pypdf import PdfReader
from src.vector_store.vector_store import VectorStore

PDF_PATH = "src/documents/SGBank_Withdrawal_Policy_and_Procedures.pdf"

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def split_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def main():
    print("Loading PDF...")
    text = load_pdf(PDF_PATH)

    print("Splitting text...")
    chunks = split_text(text)

    print("Initializing vector store...")
    vs = VectorStore(persist_directory="vectordb")

    print("Adding documents...")
    vs.add_documents(chunks)

    # vs.persist()
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
