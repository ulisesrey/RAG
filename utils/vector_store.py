from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from tqdm import tqdm
import re


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
