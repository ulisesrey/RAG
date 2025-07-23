from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import time
import logging
from datetime import datetime
from tqdm import tqdm

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

# Start timer
start = time.time()

# Load the data
pdf_path = "data/charles_darwin_origin_of_species_short.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split the documents
text_splitter= CharacterTextSplitter(
    chunk_size=2000, chunk_overlap=10, separator="\n")

split_documents = text_splitter.split_documents(documents=documents)

embeddings = OllamaEmbeddings(model="mistral")

# Batch embbedding
batch_size = 16
texts = []
vectors = []
metadatas = []

for i in tqdm(range(0, len(split_documents), batch_size), desc="Embedding batches"):
    batch_docs = split_documents[i:i + batch_size]
    batch_texts = [doc.page_content for doc in batch_docs]
    batch_metas = [doc.metadata for doc in batch_docs]

    batch_vectors = embeddings.embed_documents(batch_texts)

    texts.extend(batch_texts)
    vectors.extend(batch_vectors)
    metadatas.extend(batch_metas)

logging.info(f"Created vector store in {time.time() - start:.2f} seconds.")


# Vector Store
# vector_store = InMemoryVectorStore(embeddings) For in memory (can't be saved)
db = Chroma(
    embedding_function=embeddings,
    persist_directory="vector_store_short"
)

# Dont need db.persist() to save
db.add_texts(texts=texts, metadatas=metadatas, embeddings=vectors)



logging.info(f"Created and saved Chroma vector store in {time.time() - start:.2f} seconds.")




# if __name__ == "__main__":
#     main()
