# Document Loaders in LangChain (RAG Systems)

## What is a Document Loader?

A Document Loader in LangChain is a component responsible for loading data from external sources (files, websites, databases, etc.) and converting it into a standardized format called a `Document`.

Each `Document` typically contains:

* page_content → the actual textual data
* metadata → additional context (file name, page number, URL, etc.)

This standardization allows all downstream components in a pipeline to work seamlessly, regardless of the original data source.

---

## Why Document Loaders Matter in RAG

In a Retrieval-Augmented Generation (RAG) system, document loaders are the first and foundational component.

### RAG Pipeline Overview

```
[Document Loaders] -> [Text Splitter] -> [Embeddings] -> [Vector Database] -> [Retriever] -> [LLM]
```

### Key Role

Document loaders:

* Ingest raw data from various sources
* Convert it into a unified format
* Provide context through metadata

Without document loaders, a RAG system has no knowledge base to retrieve from.

---

## How Document Loaders Help in RAG

### 1. Data Ingestion

They allow you to bring data from:

* Local files
* Web pages
* Structured datasets

### 2. Data Normalization

Different formats (PDF, CSV, HTML) are converted into a common structure (`Document`).

### 3. Metadata Enrichment

Important context is preserved:

* Source file
* Page number
* URL
* Row index (for CSV)

### 4. Scalability

They enable ingestion from multiple sources at scale.

---

## Types of Document Loaders

### 1. TextLoader

Purpose: Load plain text files (.txt)

Use Cases:

* Notes
* Logs
* Simple documentation

Example:

```python
from langchain_community.document_loaders import TextLoader
loader = TextLoader("file.txt")
documents = loader.load()
```

Best suited for simple, unstructured text.

---

### 2. PyPDFLoader

Purpose: Load PDF documents

Key Feature:

* Extracts content page by page

Use Cases:

* Research papers
* Reports
* Resumes

Example:

```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("file.pdf")
documents = loader.load()
```

Very useful for multi-page structured documents.

---

### 3. DirectoryLoader

Purpose: Load multiple files from a folder

Key Feature:

* Works with other loaders (TextLoader, PyPDFLoader, CSVLoader, etc.)

Use Cases:

* Bulk ingestion
* Production pipelines

Example:

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    "folder_path",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()
```

Essential for scaling document ingestion.

---

### 4. WebBaseLoader

Purpose: Load content from web pages

Key Feature:

* Scrapes and parses HTML into text

Use Cases:

* Blogs
* Documentation
* Knowledge bases

Example:

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com")
documents = loader.load()
```

Useful for incorporating external knowledge.

---

### 5. CSVLoader

Purpose: Load structured CSV data

Key Feature:

* Each row becomes a separate Document

Use Cases:

* Business data
* Analytics datasets
* Tabular information

Example:

```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("data.csv")
documents = loader.load()
```

Great for integrating structured data into RAG.

---

## Summary Comparison

| Loader          | Input Type | Output Granularity | Use Case        |
| --------------- | ---------- | ------------------ | --------------- |
| TextLoader      | .txt       | Full file          | Simple text     |
| PyPDFLoader     | .pdf       | Per page           | Documents       |
| DirectoryLoader | Folder     | Multiple docs      | Bulk ingestion  |
| WebBaseLoader   | URL        | Web content        | Online data     |
| CSVLoader       | .csv       | Per row            | Structured data |

---

## Key Takeaways

* Document loaders are the entry point of RAG systems
* They convert raw data into a standard format
* They enable multi-source knowledge integration
* Choosing the right loader improves retrieval quality
* DirectoryLoader is key for scalability

---

## Final Thought

A RAG system is only as good as the data it ingests.

Document loaders define:

* What knowledge is available
* How well it is structured
* How effectively it can be retrieved

---

## Additional Useful Loaders

### 1. UnstructuredFileLoader

* Handles complex formats like PDF, DOCX, PPT
* Useful for messy or semi-structured data

### 2. JSONLoader

* Loads JSON data
* Allows extracting specific fields

### 3. NotionDBLoader

* Loads data from Notion databases
* Useful for team knowledge bases

### 4. GitHubLoader

* Loads repositories or files from GitHub
* Useful for code-based RAG

### 5. ConfluenceLoader

* Loads enterprise documentation
* Useful in corporate environments

These loaders help extend RAG systems into real-world production data sources.
