"""
The RecursiveCharacterTextSplitter intelligently divides text by prioritizing larger boundaries like paragraphs or sentences before resorting to smaller ones like spaces. It recursively ensures chunks are as meaningful as possible without exceeding size limits.

It attempts to keep larger units like paragraphs intact. If a unit exceeds the chunk size, it moves to the next level (sentences), and continues down to the word level if necessary. Langchain
The default separator hierarchy is: ["\n\n", "\n", " ", ""]
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)

long_text = """
The RecursiveCharacterTextSplitter intelligently divides text by prioritizing larger boundaries like paragraphs or sentences before resorting to smaller ones like spaces. It recursively ensures chunks are as meaningful as possible without exceeding size limits.

It attempts to keep larger units like paragraphs intact. If a unit exceeds the chunk size, it moves to the next level (sentences), and continues down to the word level if necessary. Langchain
The default separator hierarchy is: ["\n\n", "\n", " ", ""]
"""

chunks = splitter.split_text(long_text)

for chunk in chunks:
    print(f"chunk: ${chunk}")
