from langchain_openai import AzureChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import os

load_dotenv()

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    deployment_name="gpt-4o-mini",
    api_version="2025-01-01-preview"
)

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3:together",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model_hf = ChatHuggingFace(llm=llm)

parser1 = StrOutputParser()

class ReviewSentiment(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="The overall sentiment of the review.")
    summary: str = Field(description="A concise summary of the review.")

parser2 = PydanticOutputParser(pydantic_object=ReviewSentiment)

prompt1 = PromptTemplate(
    template="You are a helpful assistant. Please analyze the sentiment of the following review:\n\n {review} \n\n Provide your analysis in the following JSON format:\n{format_instructions}",
    input_variables=["review"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

print(classifier_chain.invoke({"review": "The product is amazing! I loved it and will definitely buy it again."}))

prompt2 = PromptTemplate(
    template="You are a helpful assistant. Write an appropriate response to the positive review: {review}",
    input_variables=["review"]
)

prompt3 = PromptTemplate(
    template="You are a helpful assistant. Write an appropriate response to the negative review: {review}",
    input_variables=["review"]
)

prompt4 = PromptTemplate(
    template="You are a helpful assistant. Write an appropriate response to the neutral review: {review}",
    input_variables=["review"]
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model_hf | parser1),
    (lambda x: x.sentiment == "negative", prompt3 | model_hf | parser1),
    (lambda x: x.sentiment == "neutral", prompt4 | model_hf | parser1),
    RunnableLambda(lambda x: "Sorry, I couldn't determine the sentiment of the review.")  # Default case if none of the conditions match
)

chain = classifier_chain | branch_chain

result = chain.invoke({"review": "The product is amazing! I loved it and will definitely buy it again."})

print(result)