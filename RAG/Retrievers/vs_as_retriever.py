"""
WHAT IT DOES:
  Converts the query to an embedding vector and returns the k chunks
  whose embedding vectors are closest to the query vector.
 
WHEN TO USE:
  - Default starting point for any RAG pipeline.
  - Works well when queries are clear and specific.
  - Fast, no extra LLM calls.
 
SEARCH TYPE:  "similarity"
"""

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

dir_path = os.path.dirname(__file__)
docxPath = os.path.join(dir_path, "..", "Document_Loader", "Documents", "wordMS", "novatech_knowledge_base.docx")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "novatech-kb")

def show_results(docs, label):
    print(f"\n  ── {label} ── ({len(docs)} docs returned)")
    for i, doc in enumerate(docs):
        section  = doc.metadata.get("section", "?")
        chunk_id = doc.metadata.get("chunk_id", "?")
        preview  = doc.page_content[:160].replace("\n", " ")
        print(f"  [{i+1}] chunk={chunk_id} | section={section}")
        print(f"       {preview}...")

Loader = Docx2txtLoader(docxPath)
docs = Loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["/n/n", "/n", ".", " "],
)

chunks = splitter.split_documents(docs)

# Add some useful metadata fields — Self-Query retriever will use these later
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"]  = i
    chunk.metadata["source"]    = "novatech_knowledge_base"
    # Assign a section tag based on keyword presence in chunk text
    text_lower = chunk.page_content.lower()
    if "pricing" in text_lower or "plan" in text_lower or "usd" in text_lower:
        chunk.metadata["section"] = "pricing"
    elif "security" in text_lower or "compliance" in text_lower or "gdpr" in text_lower:
        chunk.metadata["section"] = "security"
    elif "api" in text_lower or "sdk" in text_lower or "integration" in text_lower:
        chunk.metadata["section"] = "integration"
    elif "support" in text_lower or "sla" in text_lower or "incident" in text_lower:
        chunk.metadata["section"] = "support"
    elif "roadmap" in text_lower or "2025" in text_lower:
        chunk.metadata["section"] = "roadmap"
    else:
        chunk.metadata["section"] = "general"
 
print(f"  Total chunks after split  : {len(chunks)}")
print(f"  Avg chunk size (chars)    : {sum(len(c.page_content) for c in chunks) // len(chunks)}")
print(f"  Sample chunk:\n  {chunks[2].page_content[:200]}...")

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    chunk_size=16
)

vector_store = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
    index_name=INDEX_NAME,
    embedding_function=embeddings.embed_query,
    additional_search_client_options={"api_version": "2023-11-01"},
)

vector_store.add_documents(chunks)

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    deployment_name="gpt-4o-mini",
    api_version="2025-01-01-preview",
    temperature=0.34,
)

similarity_retriever = vector_store.as_retriever(
    search_type = "similarity",
    k=4,
)

query = "What are the pricing plans for NovaMind?"
results  = similarity_retriever.invoke(query)
show_results(results, f'Similarity  |  query="{query}"')

