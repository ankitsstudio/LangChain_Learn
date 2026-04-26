"""
What is PydanticOutputParser?
PydanticOutputParser is a powerful tool in the LangChain library that allows you to define a structured schema for the expected output from a language model using Pydantic, a data validation and settings management library. It provides a way to ensure that the output from the language model adheres to a specific format and structure, making it easier to work with and process the data.

Why use PydanticOutputParser?
1. Strict Schema Enforcement: Pydantic allows you to define a strict schema for the expected output, ensuring that the data adheres to the specified types and constraints.
2. Type safety: Pydantic provides type safety, which helps catch errors early in the development process and ensures that the data is in the expected format.
3. Easy Validation: Pydantic automatically validates the data against the defined schema, making it easier to handle and process the output from the language model.
4. Seamless Integration: PydanticOutputParser integrates seamlessly with LangChain, allowing you to easily incorporate structured output parsing into your language model workflows.
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3:together", 
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

class CropInfo(BaseModel):
    name: str = Field(..., description="The name of the crop.")
    type: str = Field(..., description="The type of the crop (e.g., cereal, vegetable, fruit).")
    current_price: float = Field(..., description="The current price of the crop in INR per quintal.")

parser = PydanticOutputParser(pydantic_object=CropInfo)

template = PromptTemplate(
    template="Provide me name, type and current price of a crop farmed in the {season} in area {region} of India. \n {format_instructions}",
    input_variables=["season", "region"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = template.invoke({"season": "winter", "region": "Jammu and Kashmir"})
print(prompt)

response = model.invoke(prompt)
print(response.content)

parsed_output = parser.parse(response.content)
print(parsed_output)

# chain = template | model | parser
# final_output = chain.invoke({"season": "winter", "region": "Jammu and Kashmir"})
# print(final_output)
