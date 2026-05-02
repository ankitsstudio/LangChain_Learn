from langchain_community.document_loaders import CSVLoader
import os

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "Documents", "csvDoc", "customers-100.csv")

loader = CSVLoader(file_path, encoding="utf-8")
documents = loader.load()

print(len(documents))
print(documents[12])