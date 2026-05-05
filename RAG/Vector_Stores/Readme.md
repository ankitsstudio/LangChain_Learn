# 🗃️ LangChain Vector Stores

A practical reference guide for working with vector stores in LangChain — covering setup, document ingestion, semantic search, and retrieval patterns across multiple backends.

---

## Table of Contents

- [What is a Vector Store?](#what-is-a-vector-store)
- [How It Works](#how-it-works)
- [Choosing a Vector Store](#choosing-a-vector-store)
- [Embedding Models](#embedding-models)
- [Loading & Splitting Documents](#loading--splitting-documents)
- [Vector Store Implementations](#vector-store-implementations)
  - [ChromaDB](#chromadb-local--prototyping)
  - [FAISS](#faiss-local--high-performance)
  - [Azure AI Search](#azure-ai-search-managed--production)
  - [Pinecone](#pinecone-cloud-native)
  - [Qdrant](#qdrant-self-hosted--cloud)
- [Semantic Search Patterns](#semantic-search-patterns)
- [Metadata Filtering](#metadata-filtering)
- [Full RAG Pipeline](#full-rag-pipeline)
- [Chunking Strategy Reference](#chunking-strategy-reference)
- [Common Errors & Fixes](#common-errors--fixes)

---

## What is a Vector Store?

A vector store is a database that stores text as **high-dimensional numerical vectors** (embeddings). Instead of matching keywords, it finds documents by **semantic meaning** — so a query like *"what causes rainfall"* can match a chunk that says *"precipitation occurs when water vapor condenses"* even though no words overlap.

```
"How does rainfall occur?"
        │
        ▼
  Embedding Model        ← converts query to a vector
        │
        ▼
 [0.23, -0.85, 0.41 ...]  ← 1536-dimensional float array
        │
        ▼
  Vector Store            ← finds nearest stored vectors
        │
        ▼
  Top-k similar chunks    ← returned as Document objects
```

---

## How It Works

The full pipeline from raw text to retrieval has four stages:

```
Raw Documents
     │
     ▼
Text Splitter       ← breaks large docs into chunks
     │
     ▼
Embedding Model     ← converts each chunk to a vector
     │
     ▼
Vector Store        ← stores vectors + original text + metadata
     │
     ▼
Similarity Search   ← query → vector → find nearest chunks
```

---

## Choosing a Vector Store

| Vector Store | Best For | Persistence | Cost | Hybrid Search |
|---|---|---|---|---|
| **ChromaDB** | Local dev, prototyping | Local disk | Free | No |
| **FAISS** | Speed-critical, offline | Local file | Free | No |
| **Azure AI Search** | Azure workloads, enterprise | Managed cloud | Paid | ✅ Yes |
| **Pinecone** | Cloud-native, serverless | Managed cloud | Paid (free tier) | ✅ Yes |
| **Qdrant** | Self-hosted or cloud | Both | Free / Paid | ✅ Yes |
| **Weaviate** | Multi-modal, graph-like | Both | Free / Paid | ✅ Yes |
| **PGVector** | Already using PostgreSQL | DB-managed | Free | No |

**General rule:**
- **Learning / local dev** → ChromaDB or FAISS
- **Production on Azure** → Azure AI Search
- **Production (cloud-agnostic)** → Pinecone or Qdrant
- **Already have Postgres** → PGVector

---

## Embedding Models

Embeddings convert text into vectors. The embedding model used at **query time must match** the one used at **ingestion time**.

### OpenAI Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # or text-embedding-3-large
    api_key=os.getenv("OPENAI_API_KEY"),
)
```

### Azure OpenAI Embeddings

```python
from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    chunk_size=16,  # Azure has stricter batch limits
)
```

### HuggingFace Embeddings (free, local)

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

> **Important:** `text-embedding-ada-002`, `text-embedding-3-small`, and `text-embedding-3-large` are Azure/OpenAI **deployed models** — you deploy them in your Azure subscription and reference them by deployment name, not model name directly.

---

## Loading & Splitting Documents

### Document Loaders

```python
from langchain_community.document_loaders import (
    TextLoader,           # .txt files
    PyPDFLoader,          # basic PDF (text-based only)
    PyMuPDFLoader,        # better PDF (handles complex layouts, tables)
    PDFPlumberLoader,     # best for tables in PDFs
    DirectoryLoader,      # load all files from a folder
    WebBaseLoader,        # load from a URL
    CSVLoader,            # .csv files
    UnstructuredWordDocumentLoader,  # .docx files
)

# Single PDF
loader = PyMuPDFLoader("docs/my_document.pdf")
documents = loader.load()

# All PDFs in a folder
loader = DirectoryLoader("docs/", glob="**/*.pdf", loader_cls=PyMuPDFLoader)
documents = loader.load()

# Web page
loader = WebBaseLoader("https://example.com/article")
documents = loader.load()
```

> **PDF loader selection guide:**
> - `PyPDFLoader` — simple text PDFs, fastest
> - `PyMuPDFLoader` — tables, complex layouts, mixed content ✅ recommended default
> - `PDFPlumberLoader` — best table extraction
> - If all loaders return empty content → the PDF likely has text as vector paths, use OCR (`pytesseract`)

### Text Splitters

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

# General purpose — tries to split on paragraphs, then sentences, then words
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,        # overlap preserves context at boundaries
    separators=["\n\n", "\n", ". ", " ", ""],
)

# For code-heavy documents (Python, JS, etc.)
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=150,
)

chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")
```

#### Chunk size guidelines

| Document Type | `chunk_size` | `chunk_overlap` |
|---|---|---|
| Research papers / books | 1000–1500 | 150–200 |
| Legal / contracts | 500–800 | 100–150 |
| News articles / blogs | 400–600 | 50–100 |
| Code / technical docs | 300–500 | 50 |
| Q&A / FAQs | 200–300 | 0–30 |

**Golden rule:** chunk size should match the granularity of questions you'll ask. Broad summary questions → larger chunks. Specific factual questions → smaller chunks.

---

## Vector Store Implementations

### ChromaDB (Local / Prototyping)

**Install:**
```bash
pip install langchain-chroma chromadb
```

```python
from langchain_chroma import Chroma

# Create and persist
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",    # saves to disk automatically
    collection_name="my_collection",
)

# Load existing store (no re-embedding needed)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="my_collection",
)

# Inspect
print(vectorstore._collection.count())  # number of stored chunks
```

---

### FAISS (Local / High Performance)

FAISS (Facebook AI Similarity Search) is the fastest option for local use — entirely in-memory with optional disk save. No server needed.

**Install:**
```bash
pip install faiss-cpu   # or faiss-gpu for GPU support
```

```python
from langchain_community.vectorstores import FAISS

# Create
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save to disk
vectorstore.save_local("faiss_index")

# Load from disk
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

# Merge two FAISS indexes
vectorstore.merge_from(another_faiss_store)
```

> **When to use FAISS:** You need maximum speed with no network calls, working fully offline, or embedding millions of vectors locally.

---

### Azure AI Search (Managed / Production)

Full managed vector store on Azure with hybrid search (vector + keyword), OData filtering, and auto-scaling.

**Install:**
```bash
pip install azure-search-documents==11.4.0 azure-identity azure-core
```

**`.env` setup:**
```env
AZURE_SEARCH_ENDPOINT=https://your-service.search.windows.net
AZURE_SEARCH_API_KEY=your_admin_key
AZURE_SEARCH_INDEX_NAME=my-index
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your_key
AZURE_OPENAI_VERSION=2024-02-01
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

```python
from langchain_community.vectorstores.azuresearch import AzureSearch

vectorstore = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
    embedding_function=embeddings.embed_query,
    additional_search_client_options={"api_version": "2023-11-01"},
)

# Add documents
vectorstore.add_documents(chunks)

# Hybrid search (vector + keyword — Azure exclusive)
results = vectorstore.hybrid_search("your query", k=5)

# Similarity search
results = vectorstore.similarity_search("your query", k=5)
```

> **Known issue:** `as_retriever(search_kwargs={"k": 5})` causes a `multiple values for keyword argument 'k'` bug in some `langchain-community` versions. Workaround — use `RunnableLambda` directly:
> ```python
> from langchain_core.runnables import RunnableLambda
> retriever = RunnableLambda(lambda q: vectorstore.similarity_search(q, k=5))
> ```

---

### Pinecone (Cloud Native)

Fully managed, serverless vector database. No infrastructure to manage.

**Install:**
```bash
pip install langchain-pinecone pinecone-client
```

```python
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create index (only once)
pc.create_index(
    name="my-index",
    dimension=1536,       # must match your embedding model's output dimension
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

# Create vector store
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="my-index",
)

# Load existing index
vectorstore = PineconeVectorStore(
    index_name="my-index",
    embedding=embeddings,
)
```

> **Dimension reference:** `text-embedding-3-small` → 1536, `text-embedding-3-large` → 3072, `all-MiniLM-L6-v2` → 384

---

### Qdrant (Self-hosted / Cloud)

Open-source vector database. Can run locally via Docker or as a managed cloud service.

**Install:**
```bash
pip install langchain-qdrant qdrant-client

# Run locally via Docker
docker run -p 6333:6333 qdrant/qdrant
```

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Connect to local instance
client = QdrantClient(url="http://localhost:6333")

# Or connect to Qdrant Cloud
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Create collection (only once)
client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Create vector store
vectorstore = QdrantVectorStore(
    client=client,
    collection_name="my_collection",
    embedding=embeddings,
)

vectorstore.add_documents(chunks)

# In-memory (for testing, no Docker needed)
vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    location=":memory:",
    collection_name="my_collection",
)
```

---

## Semantic Search Patterns

All vector stores share the same LangChain interface for search:

```python
query = "What is the main topic of the document?"

# 1. Basic similarity search — returns Document objects
results = vectorstore.similarity_search(query, k=5)

# 2. With relevance scores — returns (Document, score) tuples
results = vectorstore.similarity_search_with_relevance_scores(query, k=5)
for doc, score in results:
    print(f"Score: {score:.4f} | {doc.page_content[:200]}")

# 3. MMR — diverse results, avoids returning near-duplicate chunks
results = vectorstore.max_marginal_relevance_search(
    query,
    k=5,
    fetch_k=20,        # fetch 20 candidates, return 5 diverse ones
    lambda_mult=0.6,   # 0 = max diversity, 1 = max relevance
)

# 4. As a retriever (for use inside chains)
retriever = vectorstore.as_retriever(
    search_type="similarity",   # "similarity", "mmr", "similarity_score_threshold"
    search_kwargs={"k": 5},
)
docs = retriever.invoke(query)
```

---

## Metadata Filtering

Every chunk stores metadata (source file, page number, custom tags). You can filter search to specific subsets:

```python
# Add metadata when creating documents
from langchain_core.documents import Document

doc = Document(
    page_content="Some text content here.",
    metadata={
        "source": "annual_report.pdf",
        "page": 12,
        "department": "finance",
        "year": 2024,
    }
)

# ChromaDB / FAISS filtering
results = vectorstore.similarity_search(
    query,
    k=5,
    filter={"department": "finance"},
)

# Azure AI Search filtering (OData syntax)
results = vectorstore.similarity_search(
    query,
    k=5,
    filters="metadata/year eq 2024 and metadata/department eq 'finance'",
)

# Pinecone filtering
results = vectorstore.similarity_search(
    query,
    k=5,
    filter={"year": {"$eq": 2024}},
)
```

---

## Full RAG Pipeline

A complete Retrieval-Augmented Generation chain that works with any vector store:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Format retrieved chunks for the prompt
def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', '?')} | Page: {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )

# Prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:
""")

# Build the chain
retriever = RunnableLambda(lambda q: vectorstore.similarity_search(q, k=5))

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Invoke
answer = rag_chain.invoke("What is the main topic of the document?")
print(answer)
```

---

## Chunking Strategy Reference

```
chunk_size too large  →  retrieval returns less focused chunks
                         LLM gets too much noise in the context

chunk_size too small  →  retrieval returns fragments without context
                         important sentences get cut mid-thought

chunk_overlap too low →  key info at chunk boundaries gets lost

chunk_overlap too high → redundant content, slower indexing
```

**Recommended starting point for most use cases:** `chunk_size=1000`, `chunk_overlap=150`

---

## Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: azure.identity` | Missing package | `pip install azure-identity` |
| `azure-search-documents` API version conflict | SDK too new for langchain-community | `pip install azure-search-documents==11.4.0` |
| `got multiple values for keyword argument 'k'` | langchain-community AzureSearch retriever bug | Use `RunnableLambda` wrapper instead of `as_retriever` |
| `File path is not a valid file or url` | Missing file extension in path | Add `.pdf` (or correct extension) to the file path |
| `Stored 0 chunks` | PDF pages have empty content | Switch loader: try `PyMuPDFLoader` → `PDFPlumberLoader` → OCR |
| `I don't know based on the provided documents` | Query doesn't semantically match stored content | Ask a question relevant to your document's actual content; debug with a direct `similarity_search` call first |
| `PyTorch was not found` warning | Optional dependency missing | Harmless if using OpenAI/Azure embeddings — ignore it |

---

## Project Structure Suggestion

```
your-project/
│
├── RAG/
│   └── Vector_Stores/
│       ├── README.md                  ← this file
│       ├── chroma_example.py          ← local ChromaDB usage
│       ├── faiss_example.py           ← local FAISS usage
│       ├── azure_search_example.py    ← Azure AI Search usage
│       ├── pinecone_example.py        ← Pinecone usage
│       ├── full_rag_chain.py          ← end-to-end RAG pipeline
│       └── vector_store.py            ← document ingestion script
│
├── Documents/
│   └── pdfDoc/
│       └── your_document.pdf
│
├── .env                               ← API keys (never commit this)
├── .env.example                       ← template with key names only
└── requirements.txt
```

**`.env.example`** (safe to commit):
```env
OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_KEY=
AZURE_OPENAI_VERSION=
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=
AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_API_KEY=
AZURE_SEARCH_INDEX_NAME=
PINECONE_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
```

---

## Requirements

```txt
# Core
langchain
langchain-core
langchain-community
langchain-openai

# Vector Stores (install only what you use)
langchain-chroma
chromadb
faiss-cpu
langchain-pinecone
pinecone-client
langchain-qdrant
qdrant-client

# Azure
azure-search-documents==11.4.0
azure-identity
azure-core

# Document Loading
pymupdf
pdfplumber
pytesseract
pdf2image

# Utilities
python-dotenv
tiktoken
```

---

*Built while learning LangChain vector stores — covers ChromaDB, FAISS, Azure AI Search, Pinecone, and Qdrant with practical patterns and real error fixes.*