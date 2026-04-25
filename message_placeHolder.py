from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

chat_History = []

chatTemplate = ChatPromptTemplate([
    ('system', "You are a helpful assistant in the domain {domain}."),
    MessagesPlaceholder(variable_name="history"),
    ('human', "Tell me about the {topic} in simple terms.")
])

prompt = chatTemplate.invoke({"history": chat_History, "domain": "Geography", "topic": "the capital of France"})
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    deployment_name="gpt-4o-mini",
    api_version="2025-01-01-preview"
)


while True: 
    user_input = input("You: ")
    chat_History.append(("user", user_input))
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break
    prompt = chatTemplate.invoke({"history": chat_History, "domain": "Geography", "topic": user_input})
    response = model.invoke(prompt)
    chat_History.append(("ai", response.content))
    print(f"AI-Bot: {response.content}")

print("\nChat History:")
for speaker, message in chat_History:
    print(f"{speaker}: {message}")