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


embeddings = OllamaEmbeddings(model="mistral")

db = Chroma(persist_directory="vector_stores/vector_store_short",
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


query = "How many animals there are which will not breed"
docs = db.similarity_search(query)
unique_docs = deduplicate_documents(docs)

# Option: test db.max_marginal_relevance_search(query, k=10, fetch_k=20)

for i, doc in enumerate(unique_docs):
    print(f"Document {i}:, {doc.page_content} \n")
    print(f"metadata is:{doc.metadata}\n\n")