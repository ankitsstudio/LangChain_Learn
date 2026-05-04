long_text = """
The CharacterTextSplitter is the most basic splitting technique — it divides text based on a specified number of characters, making it suitable for simple, uniform text splitting tasks.

It splits based on a character separator like a newline or semicolon. It will prioritize splitting on the separator rather than strictly enforcing chunk size.
"""

from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=10
)

chunks = splitter.split_text(long_text)

for chunk in chunks:
    print(f"chunk: {chunk}")