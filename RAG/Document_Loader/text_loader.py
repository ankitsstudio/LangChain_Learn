from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableSequence, RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "doc.txt")

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    deployment_name="gpt-4o-mini",
    api_version="2025-01-01-preview"
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Summarize the following text: {text}",
    input_variables=["text"]
)

loader = TextLoader(file_path, encoding="utf-8")

def load_page_content():
    documents = loader.load()
    return documents[0].page_content

documents = load_page_content()

doc = RunnableLambda(lambda _: {"text": load_page_content()})

chain = RunnableSequence(doc | prompt | model | parser)
result = chain.invoke({})  # No input needed since the document content is loaded within the RunnableLambda
print(result)