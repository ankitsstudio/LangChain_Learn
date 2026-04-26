"""
Structured output parsing is a powerful technique that allows you to extract specific information from the output of a language model in a structured format. This can be particularly useful when you want to ensure that the output adheres to a certain schema or when you want to easily access specific pieces of information from the model's response.

The limitation of the structured output parser is that it relies on the model to follow the specified format, and if the model does not adhere to the format instructions, it can lead to parsing errors or incorrect data extraction. To mitigate this, it's important to provide clear and detailed instructions to the model and to implement error handling in your code to manage cases where the output does not match the expected structure. Data validation is not automatically performed by the structured output parser, so you may need to implement additional checks to ensure that the extracted data is valid and meets your requirements.


Currently, the structured output parser does not perform automatic data validation. It relies on the model to generate output that adheres to the specified format, but it does not validate the content of the output against any schema or rules. If you need to ensure that the extracted data is valid, you would need to implement additional validation logic in your code after parsing the output. This could involve checking for required fields, validating data types, or applying custom validation rules based on your specific use case.

Currently Structured output parser is not available to import from langchain_core.output_parsers and langchain.output_parsers.
"""
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
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

schema = [
    ResponseSchema(name="fact1", description="The first fact about the topic."),
    ResponseSchema(name="fact2", description="The second fact about the topic."),
    ResponseSchema(name="fact3", description="The third fact about the topic."),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give me 3 facts about the following {topic} \n {format_instructions}",
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser

final_output = chain.invoke({'topic': 'the Eiffel Tower'})
print(final_output)
