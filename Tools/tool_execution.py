"""
Tool execution is the step where the actual python function is run using the input arguments that the LLM suggested during the tool calling.

In simplers words, the llm says:
    "Hey, call the multiply tool with these arguments: a=5, b=6"

    Tool execution is wher you or langchain actually runs:
    multiply(a=5, b=6)
    and gets the result 30 and returns it back to the llm.
"""
from rich.pretty import pprint
from langchain.tools import tool
from langchain_openai import AzureOpenAI, ChatOpenAI, AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import requests
import os

load_dotenv()

# tool creation step
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

# tool binding step
llm_with_tools = llm.bind_tools([get_current_weather])

query = HumanMessage(content="tell me the wheather in New York")
messages = [query]

# tool calling step
result1 = llm_with_tools.invoke(messages)
messages.append(result1)
# print(result1.tool_calls)
# print(result1)
pprint(f"Messages after tool calling: {messages}")

# Tool Execution step
for tool_call in result1.tool_calls:
    if tool_call["name"] == "get_current_weather":
        tool_result = get_current_weather.invoke(tool_call["args"])
        tool_result1 = get_current_weather.invoke(tool_call)
        pprint(f"Type of tool_result1: {type(tool_result1)}")
        pprint(f"Value of tool_result1: {tool_result1}")
        messages.append(tool_result1)

pprint(f"Messages after tool execution: {messages}")

response = llm_with_tools.invoke(messages)
pprint(response)

messages.append(response)
pprint(f"Messages: {messages}")