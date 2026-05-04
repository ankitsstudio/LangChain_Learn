from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "..", "Document_Loader", "Documents", "pdfDoc", "file-example_PDF_1MB.pdf")

loader = PyPDFLoader(file_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)

chunks = splitter.split_documents(documents)

for chunk in chunks:
    print(f"Chunk: {chunk.page_content}")
    # print(f"Metadata: {chunk.metadata}")