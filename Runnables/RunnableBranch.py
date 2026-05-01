from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnablePassthrough
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
    template="write a detailed report on the topic: {topic}",
    input_variables=["topic"]
)

report_chian = RunnableSequence(prompt1 | model | parser)

prompt2 = PromptTemplate(
    template="summarize the following report: {report}",
    input_variables=["report"]
)

summarization_chain = RunnableSequence(prompt2 | model | parser)


condition_chain = RunnableBranch(
    (lambda x: len(x.split()) > 1000, summarization_chain),
    RunnablePassthrough(),
)

final_chain = RunnableSequence(report_chian | condition_chain)

result = final_chain.invoke({"topic": "The impact of AI on society"})
print(result)