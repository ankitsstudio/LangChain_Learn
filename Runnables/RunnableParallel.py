from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
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
    template="Write a haiku about the following topic: {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Write a limerick about the following topic: {topic}",
    input_variables=["topic"]
)

parallel_chain = RunnableParallel(
    haiku=RunnableSequence(prompt1 | model | parser),
    limerick=RunnableSequence(prompt2 | model | parser)
)

result = parallel_chain.invoke({"topic": "sunsets"})
print(result)