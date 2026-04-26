"""the biggest limitation of the json output parser is that the model might not always follow the format instructions and can return invalid JSON. To handle this, we can implement error handling and validation mechanisms to ensure that the output is correctly parsed and any issues are gracefully managed."""

from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3:together",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

# template = PromptTemplate(
#     template="Provide me name, age and city of residence of the fictional person. \n {format_instructions}",
#     input_variables=["format_instructions"],
#     partial_variables={'format_instructions': parser.get_format_instructions()}
# )

template = PromptTemplate(
    template="Give me 5 facts about {topic}. \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# prompt = template.format()

# Response = model.invoke(prompt)
# print("Raw Response from the model:")
# print(Response)

# parsed_output = parser.parse(Response.content)
# print(parsed_output)

chain = template | model | parser
final_output = chain.invoke({'topic': 'the Eiffel Tower'})
print(final_output)