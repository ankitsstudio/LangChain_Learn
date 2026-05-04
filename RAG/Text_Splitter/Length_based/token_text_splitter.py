long_text="""
Token-based splitting divides text based on the number of tokens, which is directly useful when working with language models since they have token-based context limits — not character-based ones.
You can also use CharacterTextSplitter.from_tiktoken_encoder() for OpenAI-compatible token counting:
"""

from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    encoding_name="cl100k_base",
    chunk_size=100,
    chunk_overlap=10
)

chunks = splitter.split_text(long_text)

for chunk in chunks:
    print(f"chunk: {chunk}")