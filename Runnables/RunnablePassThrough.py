from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    deployment_name="gpt-4o-mini",
    api_version="2025-01-01-preview"
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a joke about the following topic: {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Write the explanation of the joke about the following topic: {topic}",
    input_variables=["topic"]
)

joke_chain = RunnableSequence(prompt1 | model | parser)
explanation_chain = RunnableSequence(prompt2 | model | parser)

parallel_chain = RunnableParallel(
    joke = RunnablePassthrough(),
    explanation = explanation_chain
)

complete_chain = joke_chain | parallel_chain

result = complete_chain.invoke({"topic": "programming"})
print(f"Joke: {result['joke']}")
print(f"Explanation: {result['explanation']}")