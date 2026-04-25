from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    deployment_name="gpt-4o-mini",
    api_version="2025-01-01-preview"
)

Chat_History = []

while True:
    user_input = input("You: ")
    Chat_History.append(("user", user_input))
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break
    response = model.invoke(Chat_History)
    Chat_History.append(("ai", response.content))
    print(f"AI-Bot: {response.content}")

print("\nChat History:")
for speaker, message in Chat_History:
    print(f"{speaker}: {message}")