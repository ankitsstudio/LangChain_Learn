"""
WHAT IT DOES:
  Uses the LLM to generate N rephrased versions of the user's query,
  runs each one against the vector store independently, then merges and
  deduplicates all the results.
 
  Example: "how to connect novatech to python?"  →  LLM generates:
    1. "Python SDK installation for NovaMind"
    2. "NovaMind Python package pip install"
    3. "Integrating NovaTech API with Python code"
  Each hits the vector store separately → wider recall.
 
WHEN TO USE:
  - Users type short, vague, or ambiguous queries.
  - You want to maximise recall (at the cost of extra LLM token use).
  - Good complement to MMR — MultiQuery for breadth, MMR for diversity.
 
COST NOTE:
  Makes 1 extra LLM call per user query (for query expansion).
  Query generation is cheap — usually < 200 tokens.
"""

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
import os

load_dotenv()


INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

def show_results(docs, label):
    print(f"\n  ── {label} ── ({len(docs)} docs returned)")
    for i, doc in enumerate(docs):
        section  = doc.metadata.get("section", "?")
        chunk_id = doc.metadata.get("chunk_id", "?")
        preview  = doc.page_content.replace("\n", " ")
        print(f"  [{i+1}] chunk={chunk_id} | section={section}")
        print(f"       {preview}...")

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    chunk_size=16
)

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    deployment_name="gpt-4o-mini",
    api_version="2025-01-01-preview",
    temperature=0.34,
)

vector_store = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
    index_name=INDEX_NAME,
    embedding_function=embeddings.embed_query,
    additional_search_client_options={"api_version": "2023-11-01"},
)


simlarity_retriever = vector_store.as_retriever(
    search_type="similarity",
    k=5
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=simlarity_retriever,
    llm=llm,
    include_original=True,
)

query = "How do I get started with NovaTech?"
docs  = multi_query_retriever.invoke(query)
show_results(docs, f'MultiQuery  |  query="{query}"')