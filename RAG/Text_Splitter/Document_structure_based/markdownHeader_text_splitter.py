markdown_text = """
# What is MarkdownHeaderTextSplitter?

MarkdownHeaderTextSplitter is a document-structure-based text splitter in LangChain. It splits Markdown documents at heading boundaries, treating each section as a semantically coherent chunk.

Each resulting chunk carries metadata that records the heading hierarchy it belongs to. For instance, a chunk under "## Usage" nested inside "# Introduction" gets metadata like {"Header 1": "Introduction", "Header 2": "Usage"}.

## Why use it?

Use this splitter when your documents have a natural section structure defined by Markdown headings. It is the ideal choice for technical documentation, README files, wikis, and knowledge base articles.

Without structure-aware splitting, a generic character-based splitter might cut in the middle of a section, losing the context of the heading it belonged to.

### Advantages over CharacterTextSplitter

Unlike CharacterTextSplitter which splits blindly at character boundaries, MarkdownHeaderTextSplitter respects the document's outline. This means your vector store chunks are topically focused, which greatly improves retrieval precision.

## How to use it

Import the splitter from langchain_text_splitters. Define a list of (marker, label) tuples for the heading levels you want to split on. Then call split_text() with your raw Markdown string.

### Code example

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers = [("#", "H1"), ("##", "H2"), ("###", "H3")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
docs = splitter.split_text(markdown_text)
for doc in docs:
    print(doc.metadata)
    print(doc.page_content)
```

Each doc in the output is a LangChain Document object with page_content set to the text of that section and metadata reflecting the heading path.

## When to combine with RecursiveCharacterTextSplitter

If individual sections are very long, you can pass the output of MarkdownHeaderTextSplitter into a RecursiveCharacterTextSplitter as a second stage. This gives you both structural boundaries and size control.
"""

from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_text)

for chunk in chunks:
    print(f"Chunk: {chunk.page_content}")
    print(f"Metadata: {chunk.metadata}")