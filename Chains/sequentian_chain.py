"""
Sequentialian Chains in Langchain allow you to create a sequence of calls that are executed in order, where the output of one call can be used as the input for the next call. This is particularly useful for creating complex workflows that involve multiple steps, such as generating a response based on a prompt, parsing the output, and then using that parsed information to make another query.
"""

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os


load_dotenv()

class CapitalInfo(BaseModel):
    country: str = Field(..., description="The name of the country")
    capital: str = Field(..., description="The capital city of the country")
    landmark: str = Field(..., description="A famous landmark located in the capital city")

parser1 = PydanticOutputParser(pydantic_object=CapitalInfo)
parser2 = StrOutputParser()

prompt1 = PromptTemplate(
    template="What is the capital of {country} and what the famous landmark is located there? \n {format_instructions}",
    input_variables=["country"],
    partial_variables={"format_instructions": parser1.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template="What is the population of {capital} of {country} and what tell me 5 facts about the {landmark}?",
    input_variables=["capital", "country", "landmark"]
)

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    deployment_name="gpt-4o-mini",
    api_version="2025-01-01-preview"
)

# output of parser1 will be a CapitalInfo object, we can use that to extract the capital, country and landmark and pass it to the next prompt 
# but since the output of parser1 is a CapitalInfo object, we can use a lambda function to extract the capital, country and landmark and pass it to the next prompt
chain = (prompt1 | model | parser1 | (lambda x: { "capital": x.capital, "country": x.country, "landmark": x.landmark})| prompt2| model| parser2 )
response = chain.invoke({'country': 'Mexico'})
print(response)
