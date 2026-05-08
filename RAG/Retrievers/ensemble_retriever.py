"""
WHAT IT DOES:
  Runs two (or more) retrievers in parallel and merges their results
  using Reciprocal Rank Fusion (RRF):
    score_rrf = Σ 1 / (rank + 60)
  Each retriever's results are ranked, scores combined, final list re-ranked.
 
  BM25   → keyword matching. Great for exact terms, product names, codes.
  Vector → semantic matching. Great for conceptual/paraphrased queries.
  Together they cover both literal and semantic search.
 
WHEN TO USE:
  - Your queries mix exact terms (e.g. "NovaMind-70B") with natural language.
  - You can't predict whether users will use technical jargon or plain speech.
  - Best practice for production RAG on heterogeneous documents.
 
AZURE NOTE:
  Azure AI Search has built-in hybrid search (BM25 + vector + semantic ranker).
  The EnsembleRetriever is useful when you want to combine:
    - Azure AI Search (semantic) + a local BM25 over a smaller corpus
    - Two different Azure Search indexes
    - Azure Search + another data source entirely
"""

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from dotenv import load_dotenv
import os

load_dotenv()

INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
dir_path = os.path.dirname(__file__)
docxPath = os.path.join(dir_path, "..", "Document_Loader", "Documents", "wordMS", "novatech_knowledge_base.docx")

def show_results(docs, label):
    print(f"\n── {label} ── ({len(docs)} docs returned)")
    
    for i, doc in enumerate(docs):
        section = doc.metadata.get("section", "?")
        chunk_id = doc.metadata.get("chunk_id", "?")
        preview = doc.page_content

        print(f"[{i+1}] chunk={chunk_id} | section={section}")
        print(f"    {preview}...")

Loader = Docx2txtLoader(docxPath)
docs = Loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["/n/n", "/n", ".", " "],
)

chunks = splitter.split_documents(docs)


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

vvector_retriever = vector_store.as_retriever(
    search_type="hybrid",
    k=6,
)

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5


ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vvector_retriever],
    weights=[0.3, 0.7],
)


query = "NovaMind-70B token limit context window"
retrieved_docs  = ensemble_retriever.invoke(query)
show_results(retrieved_docs, f'Ensemble (BM25+Vector)  |  query="{query}"')