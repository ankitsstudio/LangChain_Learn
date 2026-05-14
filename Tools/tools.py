from langchain.tools import tool

# @tools is a decorator that allows you to define a function as a tool that can be used by the llm or an agent.
@tool
def add_numbers(a: int, b: int) -> int:
    """This tool adds up two numbers."""
    return a + b

# As the function is decorated with @tool, it can be invoked using the invoke method, which takes a dictionary of arguments.
result = add_numbers.invoke({"a":1, "b": 2})
print(result)

