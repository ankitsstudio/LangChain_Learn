from langchain_community.tools import DuckDuckGoSearchRun, ShellTool
# DuckDuckGoSearchRun is a tool that allows you to perform a search using the DuckDuckGo search engine.

ddgSearch_tool = DuckDuckGoSearchRun()
shell_tool = ShellTool()

result = ddgSearch_tool.invoke({"query": "What is the capital of France?"})
shell_result = shell_tool.invoke({"commands": ["echo", "Hello", "World"]})
print(result)
print(shell_result)