from rich.pretty import pprint
from langchain.tools import tool
from langchain_openai import AzureOpenAI, ChatOpenAI, AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import requests
import json

import os

load_dotenv()

@tool
def get_conversion_rate(from_currency: str, to_currency: str) -> float:
    """Get the conversion rate between two currencies"""
    url = f"https://v6.exchangerate-api.com/v6/{os.getenv('EXCHANGE_RATE_API_KEY')}/pair/{from_currency}/{to_currency}"
    response = requests.get(url)
    return response.json()


@tool
def convert_currency(amount: float, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """Convert an amount from one currency to another using the conversion rate"""
    return amount * conversion_rate

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"), # model should be a
    api_version=os.getenv("AZURE_OPENAI_VERSION")
)

llm_with_tools = llm.bind_tools([get_conversion_rate, convert_currency])
query = HumanMessage(content="what is the conversion rate from USD to EUR and convert 100 USD to EUR")

messages = [query]

result = llm_with_tools.invoke(messages)
messages.append(result)

pprint(f"Messages after tool calling: {messages}")

for tool_call in result.tool_calls:
    if tool_call["name"] == "get_conversion_rate":
        tool_result = get_conversion_rate.invoke(tool_call)
        pprint(f"Value of tool_result: {tool_result}")
        conversion_rate = json.loads(tool_result.content)['conversion_rate']
        messages.append(tool_result)
    elif tool_call["name"] == "convert_currency":
        tool_call["args"]["conversion_rate"] = conversion_rate
        tool_result = convert_currency.invoke(tool_call)
        pprint(f"Value of tool_result: {tool_result}")
        messages.append(tool_result)

# pprint(f"Messages after tool execution: {messages}")

final_response = llm_with_tools.invoke(messages)
pprint(f"Final response: {final_response.content}")
    