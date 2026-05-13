"""
Tool binding: is the step where you register tools with a language model (LLM) or an agent, allowing them to access and utilize the functionalities provided by those tools. This process involves associating the tools with the LLM or agent so that they can be invoked when needed during interactions.
   1. The llm knows what toold are available 
   2. It knows what each tool does (description)
   3. It knows what input format to use when invoking the tool (args_schema)

"""
from langchain.tools import tool
from langchain_openai import AzureOpenAI, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import requests
import os

load_dotenv()

@tool
def get_current_weather(location: str) -> str:
    """Get the current weather in a given location"""
    # For the sake of this example, we'll just return a dummy weather report.
    # In a real implementation, you would make an API call to a weather service here.
    return f"The current weather in {location} is sunny with a temperature of 25°C."

llm = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"), # model should be a llm model, not a chat model
    api_version=os.getenv("AZURE_OPENAI_VERSION")
)

llm_chat_openAi = ChatOpenAI()

llm_with_tools = llm_chat_openAi.bind_tools([get_current_weather])