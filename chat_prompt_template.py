from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

chat_template = ChatPromptTemplate([
    ('system', "You are a helpful {domain} assistant."),
    ('human', "Explain the concept of {concept} in simple terms.")
])

prompt = chat_template.invoke({"domain": "math", "concept": "calculus"})

# print(prompt)

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    deployment_name="gpt-4o-mini",
    api_version="2025-01-01-preview"
)

response = model.invoke(prompt)
print(response.content)
