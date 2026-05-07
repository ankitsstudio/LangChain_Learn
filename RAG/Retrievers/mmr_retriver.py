from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.vectorstores.utils import maximal_marginal_relevance

from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()

INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

def show_results(docs, label):
    print(f"\n── {label} ── ({len(docs)} docs returned)")
    
    for i, doc in enumerate(docs):
        section = doc.metadata.get("section", "?")
        chunk_id = doc.metadata.get("chunk_id", "?")
        preview = doc.page_content[:160].replace("\n", " ")

        print(f"[{i+1}] chunk={chunk_id} | section={section}")
        print(f"    {preview}...")


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
    additional_search_client_options={
        "api_version": "2023-11-01"
    },
)

query = "What are the pricing plans for NovaMind?"


# STEP 1 -> fetch larger candidate pool
candidate_docs = vector_store.similarity_search(
    query,
    k=10
)


# STEP 2 -> query embedding
query_embedding = embeddings.embed_query(query)


# STEP 3 -> document embeddings
doc_embeddings = [
    embeddings.embed_query(doc.page_content)
    for doc in candidate_docs
]


# STEP 4 -> apply MMR
selected_indices = maximal_marginal_relevance(
    np.array(query_embedding),
    doc_embeddings,
    k=4,
    lambda_mult=0.5,
)


# STEP 5 -> final docs
mmr_docs = [candidate_docs[i] for i in selected_indices]


show_results(mmr_docs, f'Manual MMR | query="{query}"')