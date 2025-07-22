from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import time
import logging
from datetime import datetime

# Set up logging
log_id = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/{log_id}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# start timer
start = time.time()

# Load the data
pdf_path = "data/charles_darwin_origin_of_species.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split the documents
text_splitter= CharacterTextSplitter(
    chunk_size=2000, chunk_overlap=10, separator="\n")

split_documents = text_splitter.split_documents(documents=documents)

embeddings = OllamaEmbeddings(model="mistral")

# Vector Store
# vector_store = InMemoryVectorStore(embeddings) For in memory (can't be saved)
db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="vector_store_long"
)

end = time.time()

logging.info(f"Created and saved Chroma vector store in {end - start:.2f} seconds.")




# vector_store.add_documents(split_documents)

# vector_store.save_local("test")



# def main():
#     llm = ChatOllama(model="mistral:latest")
#     result = llm.invoke("What is the capital of France?")
#     print(result.content)



# if __name__ == "__main__":
#     main()
