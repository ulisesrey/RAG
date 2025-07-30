from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb
import time
import logging
from datetime import datetime
from tqdm import tqdm
from langchain import hub
from dotenv import load_dotenv
from utils.utils import load_config
from utils.retriever import deduplicate_documents, format_sources



load_dotenv()
# Load parameters from config.yaml
config = load_config()

model = config["embedding_model"]["name"]+":"+config["embedding_model"]["version"]
embeddings = OllamaEmbeddings(model=model)

persist_directory = config["persist_directory"]+config["embedding_model"]["name"]

db = Chroma(persist_directory=persist_directory,
            embedding_function=embeddings)

# Check number of docs
print("Docs stored:", db._collection.count())

query = input("Type your query:\n")#"Which is our oldest cultivated plant?"
retriever = db.as_retriever(
    search_type="mmr",  # or "similarity"
    search_kwargs={"k": 5, "fetch_k": 20}
)

docs = retriever.invoke(query)
unique_docs = deduplicate_documents(docs)
# Option: test db.max_marginal_relevance_search(query, k=10, fetch_k=20)

# Format documents with metadata
context_text = format_sources(unique_docs)

print(context_text)

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOllama(model=config["chat_model"], temperature=0.0) 

example_messages = prompt.invoke(
    {"context": context_text, "question": query}
).to_messages()

response = llm.invoke(example_messages)
print(f"This is the answer:\n {response.content}")