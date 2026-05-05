from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber
import os
from dotenv import load_dotenv

load_dotenv()

# Embedding
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    chunk_size=16
)

# print(f"searh api key: {os.getenv("AZURE_SEARCH_API_KEY")}")
# print(f"search endpoint: {os.getenv("AZURE_SEARCH_ENDPOINT")}")

# connect to azure ai search
vectorStore = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
    embedding_function=embeddings.embed_query,
    additional_search_client_options={"api_version": "2023-11-01"},
)

# load and split your document
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "..", "Document_Loader", "Documents", "pdfDoc", "ncert-textbook-for-class-10-english-footprints-without-feet-chapter-1.pdf")

# loader = PyPDFLoader(file_path)
loader = PyMuPDFLoader(file_path)
# Add this after loader.load() to inspect what was actually extracted
documents = loader.load()
print(f"Length of document: {len(documents)}")

# Check if text is actually being extracted
for i, doc in enumerate(documents[:3]):   # check first 3 pages
    print(f"\n--- Page {i+1} ---")
    print(f"Content length: {len(doc.page_content)}")
    print(f"Content preview: {doc.page_content[:200]}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)

# store in azure ai search
vectorStore.add_documents(chunks)
print(f"Stored {len(chunks)} chunks in azure ai search"),


query = "What are the key findings of the document?"

results = vectorStore.similarity_search(query, k=5)

for i, doc in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(f"Source : {doc.metadata.get('source')}")
    print(f"Content: {doc.page_content[:300]}")