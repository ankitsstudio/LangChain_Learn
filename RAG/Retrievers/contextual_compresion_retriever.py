"""
WHAT IT DOES:
  After the base retriever returns chunks, a compressor reviews each one
  and extracts ONLY the sentences/phrases relevant to the query.
  Irrelevant content within a chunk is removed before passing to the LLM.
 
  Example: A 400-token chunk about "security, pricing, and data retention"
  when the query is only about "data retention" → compressor returns only
  the 2 sentences about data retention.
 
  Two compressor options:
  ┌──────────────────────┬─────────────────────────────────────────────────┐
  │ LLMChainExtractor    │ Uses LLM to extract relevant sentences.         │
  │                      │ High quality, but costs tokens per chunk.       │
  ├──────────────────────┼─────────────────────────────────────────────────┤
  │ EmbeddingsFilter     │ Drops chunks below a similarity score threshold.│
  │                      │ No LLM cost — uses embedding comparison only.   │
  └──────────────────────┴─────────────────────────────────────────────────┘
 
WHEN TO USE:
  - Documents have long, dense chunks with mixed topics.
  - You want to reduce token usage in the final LLM call.
  - Precision matters more than recall.
"""

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_classic.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter
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

retriever = vector_store.as_retriever(search_type="hybrid", k=6)

# option - A : LLM-based extractor (sentence-level precision) ──────────────

llm_compressor = LLMChainExtractor.from_llm(llm)

compressor_retriever = ContextualCompressionRetriever(
    base_compressor=llm_compressor,
    base_retriever=retriever
)


query = "What is the data retention policy for customer data?"
result  = compressor_retriever.invoke(query)
show_results(result, f'Compression (LLM)  |  query="{query}"')


# option B: Embeddings-based filter (cost-efficient) ───────────────────

embedding_compressor = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.75,
)

embedding_retriever = ContextualCompressionRetriever(
    base_compressor=embedding_compressor,
    base_retriever=retriever,
)

result  = compressor_retriever.invoke(query)
show_results(result, f'Embedding based Compressor (LLM)  |  query="{query}"')