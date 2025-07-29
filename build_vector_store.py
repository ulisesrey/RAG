from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import time
import logging
from datetime import datetime
from tqdm import tqdm
import re
import yaml
from dotenv import load_dotenv

load_dotenv()

# Load parameters from config.yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Set up logging
log_id = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/{log_id}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)

def load_and_clean_data(path):
    # Load the data
    loader = PyPDFLoader(path) #, mode="single")#, pages_delimiter="\n\x0c")
    documents = loader.load()
    
    # Clean
    for doc in documents:
        # Replace "\n\x0c" for " "
        doc.page_content = re.sub(r"\n\x0c", " ", doc.page_content)
        # Replace \n not preceded OR followed by whitespace
        doc.page_content = re.sub(r"(?<!\s)\n(?!\s)", " ", doc.page_content)
    # Replace 
    return documents


def batch_vectorize_pdf(path, chunk_size, chunk_overlap, embeddings, batch_size=16):
    """
    Vectorize the input pdf
    """

    documents = load_and_clean_data(path)


    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, keep_separator="end", separators=["\n", "."])

    split_documents = text_splitter.split_documents(documents=documents)

    # Batch embbedding
    batch_size = batch_size
    texts = []
    vectors = []
    metadatas = []

    for i in tqdm(range(0, len(split_documents), batch_size), desc="Embedding batches"):
        batch_docs = split_documents[i : i + batch_size]
        batch_texts = [doc.page_content for doc in batch_docs]
        batch_metas = [doc.metadata for doc in batch_docs]

        batch_vectors = embeddings.embed_documents(batch_texts)

        texts.extend(batch_texts)
        vectors.extend(batch_vectors)
        metadatas.extend(batch_metas)

    return texts, vectors, metadatas


def save_text(texts, filename="output_chunks.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for i, text in enumerate(texts, 1):
            f.write(f"--- Chunk {i} ---\n")
            f.write(text.strip() + "\n\n")  # Add spacing between chunks


def save_vectors(texts, vectors, metadatas, embeddings, persist_directory):
    """
    Save the texts, vectors and medatada with Chroma
    """
    # Vector Store
    db = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    # Dont need db.persist() to save
    db.add_texts(texts=texts, metadatas=metadatas, embeddings=vectors)

    return db


if __name__ == "__main__":
    # Start timer
    start = time.time()

    path = config["pdf_path"]
    persist_directory = config["persist_directory"]
    chunk_size, chunk_overlap = config["chunk"]["size"], config["chunk"]["overlap"]
    batch_size = config["batch_size"]
    embeddings = OllamaEmbeddings(model=config["embedding_model"])


    texts, vectors, metadatas = batch_vectorize_pdf(
        path, chunk_size, chunk_overlap, embeddings, batch_size
    )
    logging.info(f"Created vector store in {time.time() - start:.2f} seconds...")
    logging.info(f"...Now proceeding to save it in {persist_directory}...")

    save_text(texts, filename="vector_stores/texts_long.txt")

    db = save_vectors(texts, vectors, metadatas, embeddings, persist_directory)

    logging.info(
        f"Saved Chroma vector store in {time.time() - start:.2f} seconds."
    )
