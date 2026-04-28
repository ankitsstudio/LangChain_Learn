"""
In Langchain a chain is a sequence of calls that are executed in order. Each call can be a prompt, a model invocation, or an output parser. Chains allow you to combine multiple steps into a single workflow, making it easier to manage complex interactions with language models.
"""

from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template="What is the capital of {country} and {state}?",
    input_variables=["country", "state"]
)

parser = StrOutputParser()

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    deployment_name="gpt-4o-mini",
    api_version="2025-01-01-preview"
)

chain = prompt | model | parser

response = chain.invoke({'country': 'India', "state": 'Rajasthan'})
print(response)

