from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv

load_dotenv()

url = "https://www.zepto.com/pn/tomato-local/pvid/7e261768-88d6-4cbb-8b9b-8718625577bd"
loader = WebBaseLoader(url)

documents = loader.load()
print(len(documents))
print(documents[0])