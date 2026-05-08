"""
WHAT IT DOES:
  Queries Wikipedia's API directly and returns the top_k most relevant
  article snippets as LangChain Document objects — exactly like any other
  retriever. No embedding, no indexing, no infrastructure needed.
 
WHEN TO USE:
  - General knowledge questions not covered by your own corpus.
  - Prototype / low-cost research tool.
  - As a fallback when your vector store returns nothing useful.
 
INSTALL:
  pip install wikipedia
 
LIMITATIONS:
  - Returns summaries / snippets, not full articles by default.
  - Not suitable for real-time or breaking-news queries.
  - No metadata filtering.
"""
 
from langchain_community.retrievers import WikipediaRetriever, TavilySearchAPIRetriever
from dotenv import load_dotenv
import time
import os

load_dotenv()

# doesn't work properly, can't hit too many free api call
# wikiRetriever = WikipediaRetriever(
#     top_k_results=4,
#     doc_content_chars_max=3000,
#     lang="en",
# )


# working with the api key
wikiRetriever = TavilySearchAPIRetriever(
    k=4, 
    api_key=os.getenv("TAVILY_API_KEY")
)

query = "what is Retrieval augmented generation in AI?"
docs = wikiRetriever.invoke(query)

print(f"\n  ── Wikipedia  |  query='{query}'  ({len(docs)} articles) ──")
for i, doc in enumerate(docs):
    title   = doc.metadata.get("title", "Unknown")
    preview = doc.page_content[:200].replace("\n", " ")
    print(f"  [{i+1}] Title: {title}")
    print(f"       {preview}...")

# time.sleep(5) 

query2 = "Large language Models transformers architecture"
docs2 = wikiRetriever.invoke(query2)

print(f"\n  ── Wikipedia  |  query='{query2}'  ({len(docs2)} articles) ──")
for i, doc in enumerate(docs2):
    title   = doc.metadata.get("title", "Unknown")
    preview = doc.page_content[:200].replace("\n", " ")
    print(f"  [{i+1}] Title: {title}")
    print(f"       {preview}...")