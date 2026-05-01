from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

def word_count(text: str) -> int:
    return len(text.split())

runnableWordCount = RunnableLambda(word_count)

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

joke_gen_chain = RunnableSequence(prompt1 | model | parser)

explanation_chain = RunnableSequence(prompt2 | model | parser)

Parallel_chain = RunnableParallel(
    joke = RunnablePassthrough(),
    word_count = RunnableSequence(joke_gen_chain | runnableWordCount),
    explanation = RunnableSequence(prompt2 | model | parser)
)

final_chain = RunnableSequence(joke_gen_chain | Parallel_chain)

result = final_chain.invoke({"topic": "AI"})
print(f"Joke: {result['joke']}")
print(f"Word Count: {result['word_count']}")
print(f"Explanation: {result['explanation']}")

