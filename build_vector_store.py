from langchain_ollama import OllamaEmbeddings
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

# local imports
from utils.vector_store import batch_vectorize_pdf, save_text, save_vectors
from utils.utils import load_config

# Set up logging
log_id = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/{log_id}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)

def main():
    # load env variables
    load_dotenv()

    # Load parameters from config.yaml
    config = load_config()
    # Start timer
    start = time.time()

    # Load variables from config
    path = config["pdf_path"]
    persist_directory = config["persist_directory"]
    texts_filename = config["texts_filename"]
    chunk_size, chunk_overlap = config["chunk"]["size"], config["chunk"]["overlap"]
    batch_size = config["batch_size"]
    embeddings = OllamaEmbeddings(model=config["embedding_model"])


    texts, vectors, metadatas = batch_vectorize_pdf(
        path, chunk_size, chunk_overlap, embeddings, batch_size
    )
    logging.info(f"Created vector store in {time.time() - start:.2f} seconds...")
    logging.info(f"...Now proceeding to save it in {persist_directory}...")

    save_text(texts, filename=texts_filename)

    save_vectors(texts, vectors, metadatas, embeddings, persist_directory)

    logging.info(
        f"Saved Chroma vector store in {time.time() - start:.2f} seconds."
    )

if __name__ == "__main__":
    main()

