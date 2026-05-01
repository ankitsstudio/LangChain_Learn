from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import os

current_dir = os.path.dirname(__file__)
dir_path = os.path.join(current_dir, "Documents", "pdfDoc")

loader = DirectoryLoader(
    path=dir_path,
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

documents = loader.load()
print(documents[23])