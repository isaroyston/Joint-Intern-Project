from vector_store.vector_store import VectorStore
from utils.pdf_loader import load_pdf
from utils.text_splitter import split_text

pdf_path = "src/documents/SGBank_Withdrawal_Policy_and_Procedures.pdf"

text = load_pdf(pdf_path)
chunks = split_text(text, chunk_size=800, overlap=100)

vs = VectorStore(persist_directory="vectordb")
vs.add_documents(chunks)
vs.persist()
