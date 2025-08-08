import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain import hub
import yaml
from dotenv import load_dotenv

load_dotenv()

# Load parameters from config.yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)

@st.cache_resource
def load_vectorstore():
    model=config["embedding_model"]["name"]+":"+config["embedding_model"]["version"]
    embeddings = OllamaEmbeddings(model=model)
    return Chroma(
        persist_directory = config["persist_directory"]+config["embedding_model"]["name"],
        embedding_function=embeddings
    )

@st.cache_resource
def load_llm():
    return ChatOllama(model=config["chat_model"], temperature=0.0)

@st.cache_resource
def load_prompt_template():
    return hub.pull("rlm/rag-prompt")

def format_sources_for_display(docs):
    """Formats each document for display in an expander, showing a preview."""
    formatted_docs = []
    for i, doc in enumerate(docs):
        metadata = doc.metadata
        source = metadata.get('source', 'unknown')
        page = metadata.get('page', 'unknown')
        content = doc.page_content.strip()
        
        # Create a short preview of the content
        preview = content[:200] + "..." if len(content) > 200 else content

        formatted_docs.append({
            "title": f"**Source: {source}, Page: {page}**",
            "preview": preview,
            "full_content": content
        })
    return formatted_docs

def format_sources_for_llm(docs):
    """Formats the full content of all documents for the LLM context."""
    return "\n\n".join([
        f"**[Source: {doc.metadata.get('source', 'unknown')}, Page: {doc.metadata.get('page', 'unknown')}]**\n{doc.page_content.strip()}"
        for doc in docs
    ])

# Streamlit UI
st.title("📚 PDF Q&A with Retrieval (RAG)")

st.markdown("Ask a question based on the indexed document.")

query = st.text_input("Enter your question:", "")

if query:
    with st.spinner("Retrieving context..."):
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10}
        )
        docs = retriever.get_relevant_documents(query)

        context_text = format_sources_for_llm(docs)
        formatted_docs_for_display = format_sources_for_display(docs)

        llm = load_llm()
        prompt = load_prompt_template()

        messages = prompt.invoke({"context": context_text, "question": query}).to_messages()

        response = llm.invoke(messages)

    st.success("Answer:")
    st.markdown(response.content)

    # with st.expander("📄 Retrieved Source Chunks"):
    #     st.markdown(context_text)
    for doc in formatted_docs_for_display:
        with st.expander(doc["title"]):
            st.markdown(doc["full_content"])
