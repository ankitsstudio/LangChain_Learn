from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "Documents", "pdfDoc", "Ankit_Kumar_AI_resume124.pdf")

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    deployment_name="gpt-4o-mini",
    api_version="2025-01-01-preview"
)

parser = StrOutputParser()

loader = PyPDFLoader(file_path)

def load_pdf():
    return loader.load()

documents = RunnableLambda(lambda _: {"documents": load_pdf()})

prompt = PromptTemplate(
    template="Summarize the content of the following PDF: {documents}",
    input_variables=["text"]
)

chain = documents | prompt | model | parser

result = chain.invoke({})
print(result)
