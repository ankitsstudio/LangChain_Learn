"""
A custom tool is a tool that you define youreself.
Use them when: 
    - you want to call your own APIs
    - you want to encapsulate the business logic
    - you want the llm to interact with your database, process data, or perform any other custom operations.

Ways to create custom tools:
    1. Using the @tool decorator: You can define a function and decorate it with @tool to make it a tool that can be invoked by the llm or an agent.
    2. Using StructuredTool & Pydantic: A structured tool in langchain is a special type of tool where the input to the tool follows a structured format defined by a Pydantic model. This allows for better validation and parsing of the input data.
    3. Using BaseTool: You can create a custom tool by inheriting from the BaseTool class and implementing the necessary methods.
"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class multiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to multiply")
    b: int = Field(required=True, description="The second number to multiply")

# def multiply_func(a: int, b: int) -> int:
#     """This function multiplies two numbers."""
#     return a * b

# multiply_tool = StructuredTool(
#     name="multiply",
#     description="This tool multiplies two numbers.",
#     args_schema=multiplyInput,
#     func=multiply_func
# )

# result = multiply_tool.invoke({"a": 3, "b": 4})

# print(result)
# print(multiply_tool.name)
# print(multiply_tool.description)


from langchain_core.tools import BaseTool
from typing import Type

class multiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "This tool multiplies two numbers."
    args_schema: Type[BaseModel] = multiplyInput

    def _run(self, a: int, b: int) -> int:
        """This function multiplies two numbers."""
        return a * b
    
multiply_tooll = multiplyTool()
result = multiply_tooll.invoke({"a": 5, "b": 6})

print(result)
print(multiply_tooll.name)
print(multiply_tooll.description)
print(multiply_tooll.args_schema)

