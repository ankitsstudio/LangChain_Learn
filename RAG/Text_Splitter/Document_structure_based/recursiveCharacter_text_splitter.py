python_code='''
"""
RecursiveCharacterTextSplitter.from_language() — Code Splitter

This splitter uses language-aware separators to keep logical units
like functions and classes together. It is a convenience constructor
on top of RecursiveCharacterTextSplitter.

Supported languages include: PYTHON, JS, TS, JAVA, C, CPP, GO,
RUBY, RUST, SCALA, MARKDOWN, LATEX, HTML, SOL, and more.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


def explain_code_splitter():
    """
    Returns a short explanation of the code splitter.
    The from_language() class method populates the separators
    list with language-specific tokens so that the splitter
    prefers to break at function or class boundaries.
    """
    explanation = (
        "The code splitter avoids cutting a function in half. "
        "It first tries to split at class definitions, then at "
        "function definitions, then at blank lines, and finally "
        "at single newlines. This hierarchy ensures that the "
        "smallest possible unit that fits in chunk_size is chosen."
    )
    return explanation


def create_python_splitter(chunk_size=500, chunk_overlap=50):
    """
    Factory function that returns a Python-aware text splitter.

    Parameters
    ----------
    chunk_size : int
        Maximum number of characters per chunk.
    chunk_overlap : int
        Number of overlapping characters between adjacent chunks.

    Returns
    -------
    RecursiveCharacterTextSplitter
        A splitter configured with Python-specific separators.
    """
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter


class CodeSplitterDemo:
    """
    Demonstration class showing how to split a Python source file
    and inspect the resulting chunks.
    """

    def __init__(self, source_code: str):
        self.source_code = source_code
        self.splitter = create_python_splitter()

    def run(self):
        chunks = self.splitter.split_text(self.source_code)
        for i, chunk in enumerate(chunks):
            print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
            print(chunk)
        return chunks

    def summary(self):
        chunks = self.run()
        return {
            "total_chunks": len(chunks),
            "avg_size": sum(len(c) for c in chunks) // len(chunks)
        }
'''
    

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Supported: PYTHON, JS, TS, JAVA, C, CPP, GO, RUBY, RUST, SCALA, MARKDOWN, LATEX, HTML...
py_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50
)
chunks = py_splitter.split_text(python_code)

for chunk in chunks:
    print(f"Chunk: {chunk}")