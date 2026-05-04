html_content = """
<html>
<body>
<h1>HTMLHeaderTextSplitter — Overview</h1>
<p>HTMLHeaderTextSplitter is a document-structure-based splitter that works on HTML content. It identifies heading tags such as h1, h2, and h3, and uses them as natural split boundaries.</p>
<p>This is particularly useful when you scrape web pages using WebBaseLoader or BSHTMLLoader and want to preserve the section structure in your chunks.</p>

<h2>How it differs from MarkdownHeaderTextSplitter</h2>
<p>While both splitters use headings to define chunks, HTMLHeaderTextSplitter parses an actual HTML DOM tree. This means it correctly handles nested tags, inline formatting like bold and italic, and attributes on heading elements.</p>
<p>MarkdownHeaderTextSplitter, on the other hand, works on raw Markdown strings and relies on the presence of # symbols.</p>

<h3>Handling nested tags</h3>
<p>When a paragraph contains child tags like strong or em, the splitter extracts only the visible text content. Attributes and styling are discarded.</p>

<h2>Installation and usage</h2>
<p>Install langchain_text_splitters using pip. Then import HTMLHeaderTextSplitter and provide a list of header tag and metadata key pairs.</p>

<h3>Basic usage example</h3>
<p>You define headers_to_split_on as a list of tuples. Each tuple maps an HTML tag name to a metadata label. The splitter then groups all content between two successive headers of the chosen levels into a single Document chunk.</p>

<h2>Combining with other splitters</h2>
<p>Because HTML pages can contain very long sections, HTMLHeaderTextSplitter is often used as a first-pass structural splitter. Its output is then fed into RecursiveCharacterTextSplitter to enforce a maximum chunk size. This two-stage approach gives you structural coherence and size control together.</p>
</body>
</html>
"""

from langchain_text_splitters import HTMLHeaderTextSplitter

headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2")]
splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(html_content)

for chunk in chunks:
    print(f"Chunk: {chunk.page_content}")
    print(f"Metadata: {chunk.metadata}")