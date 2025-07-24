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
import yaml
from dotenv import load_dotenv

load_dotenv()
# Load parameters from config.yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)

embeddings = OllamaEmbeddings(model=config["embedding_model"])

db = Chroma(persist_directory="vector_store/vector_store_test",
            embedding_function=embeddings)


def deduplicate_documents(docs):
    """deduplicate docs in case the similarity search returns more than once the same document"""
    seen = set()
    unique_docs = []
    for doc in docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique_docs.append(doc)
    return unique_docs


query = "Which is our oldest cultivated plant?"
retriever = db.as_retriever(
    search_type="mmr",  # or "similarity"
    search_kwargs={"k": 5, "fetch_k": 20}
)

docs = retriever.invoke(query)
unique_docs = deduplicate_documents(docs)
# Option: test db.max_marginal_relevance_search(query, k=10, fetch_k=20)

# Format documents with metadata
context_text = "\n\n".join([
    f"[Source: {doc.metadata.get('source', 'unknown')}] Page: {doc.metadata.get('page', 'unknown')}\n{doc.page_content.strip()}"
    for doc in unique_docs
])

print(context_text)

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOllama(model="mistral", temperature=0.0) 

example_messages = prompt.invoke(
    {"context": context_text, "question": query}
).to_messages()

response = llm.invoke(example_messages)
print(response.content)