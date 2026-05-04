json_data={
  "splitter": {
    "name": "RecursiveJsonSplitter",
    "category": "document-structure-based",
    "description": "Recursively splits a JSON object into smaller chunks by walking the key hierarchy. Nested objects are kept together as long as they fit within max_chunk_size. When an object is too large it drills into sub-keys.",
    "use_cases": [
      "Chunking large API responses",
      "Splitting configuration files",
      "Indexing nested data records",
      "Embedding structured documents"
    ]
  },
  "parameters": {
    "max_chunk_size": {
      "type": "integer",
      "default": 2000,
      "description": "Maximum number of characters in each output chunk when serialized to JSON string."
    },
    "convert_lists": {
      "type": "boolean",
      "default": False,
      "description": "If true, list values are converted to dictionaries with integer keys before splitting so they can be split like objects."
    }
  },
  "usage_example": {
    "import": "from langchain_text_splitters import RecursiveJsonSplitter",
    "init": "splitter = RecursiveJsonSplitter(max_chunk_size=300)",
    "split": "chunks = splitter.split_json(json_data)",
    "note": "Output is a list of plain Python dicts, not Document objects. Use create_documents() if you need Document objects with metadata."
  },
  "comparison": {
    "vs_character_splitter": "CharacterTextSplitter treats JSON as raw text and can split inside a key name or value. RecursiveJsonSplitter only splits at key boundaries, so each chunk is always valid JSON.",
    "vs_recursive_character": "RecursiveCharacterTextSplitter can also handle JSON text but is not aware of the structure. RecursiveJsonSplitter is structure-aware and produces semantically grouped chunks."
  }
}


from langchain_text_splitters import RecursiveJsonSplitter

splitter = RecursiveJsonSplitter(max_chunk_size=300)
chunks = splitter.split_json(json_data)

for chunk in chunks:
    print(f"{chunk}")