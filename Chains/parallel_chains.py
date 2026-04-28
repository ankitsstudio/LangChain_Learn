from langchain_openai import AzureChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel
import os

load_dotenv()

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    deployment_name="gpt-4o-mini",
    api_version="2025-01-01-preview"
)

llm1 = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3:together",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

llm2 = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3:together",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model1= ChatHuggingFace(llm=llm1)
model2= ChatHuggingFace(llm=llm2)


prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="You are a helpful assistant. Make Notes for the following text: {topic}"
)

prompt2 = PromptTemplate(
    input_variables=["topic"],
    template="You are a helpful assistant. Make a 5 question quiz for the following text: {topic}"
)

prompt3 = PromptTemplate(
    input_variables=["notes", "quiz"],
    template="Merge the following notes and quiz into a single response:\n Notes: {notes}\n Quiz: {quiz}"
)

parser = StrOutputParser()

notes_chain = prompt1 | model1 | parser
quiz_chain = prompt2 | model2 | parser

parallel_chain = RunnableParallel(
    notes = notes_chain,
    quiz = quiz_chain
)

merged_chain =  prompt3 | model | parser

chain = parallel_chain | merged_chain

result = chain.invoke({"topic": "The capital of France is Paris. It is famous for its rich history, culture, art, and architecture. Paris is home to iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also known for its vibrant culinary scene, fashion industry, and romantic ambiance."})

print(result)
