"""
Tool Calling is the process where the llm decides, during a conversation or task that it needs to use a specific tool and generates the structured output with 
   . the name of the tool it wants to use
   . and the arguments to call it with

The llm does not actually execute the tool, it just generates the output that specifies which tool to use and with what arguments. The actual execution of the tool is handled by the framework like langchain or by you, which takes care of calling the tool with the specified arguments and returning the result back to the llm.
"""

from langchain.tools import tool
from langchain_openai import AzureOpenAI, ChatOpenAI, AzureChatOpenAI
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

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"), # model should be a llm model, not a chat model
    api_version=os.getenv("AZURE_OPENAI_VERSION")
)

llm_with_tools = llm.bind_tools([get_current_weather])

# result = llm_with_tools.invoke("Hi, how are you?")
# print(result)

result1 = llm_with_tools.invoke("tell me the wheather in New York")
print(result1.tool_calls)
# print(result1)