from langchain_openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"), # model should be a llm model, not a chat model
    api_version=os.getenv("AZURE_OPENAI_VERSION")
)

response = llm.invoke("What is the capital of France?")
print(response)