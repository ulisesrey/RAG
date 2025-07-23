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

def batch_vectorize_pdf(path, chunk_size, chunk_overlap, embeddings, batch_size=16):
    """
    Vectorize the input pdf
    """

    # Load the data
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split the documents
    text_splitter= CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="\n")

    split_documents = text_splitter.split_documents(documents=documents)

    # embeddings = OllamaEmbeddings(model="mistral")

    # Batch embbedding
    batch_size = batch_size
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
    
    return texts, vectors, metadatas


def save_vectors(texts, vectors, metadatas, embeddings, persist_directory):
    """
    Save the texts, vectors and medatada with Chroma
    """
    # Vector Store
    # vector_store = InMemoryVectorStore(embeddings) For in memory (can't be saved)
    db = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    # Dont need db.persist() to save
    db.add_texts(texts=texts, metadatas=metadatas, embeddings=vectors)



if __name__ == "__main__":

    # Start timer
    start = time.time()


    path = "data/charles_darwin_origin_of_species_short.pdf"
    chunk_size, chunk_overlap = 2000, 100
    batch_size=16  
    embeddings = OllamaEmbeddings(model="mistral")
    texts, vectors, metadatas = batch_vectorize_pdf(path, chunk_size, chunk_overlap, embeddings, batch_size)
    logging.info(f"Created vector store in {time.time() - start:.2f} seconds.")

    persist_directory="vector_store_short"
    save_vectors(texts, vectors, metadatas, embeddings, persist_directory)

    logging.info(f"Created and saved Chroma vector store in {time.time() - start:.2f} seconds.")
