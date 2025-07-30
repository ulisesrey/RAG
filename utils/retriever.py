"""Util functions for the retriever"""

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

def format_sources(docs) -> str:
    """Format documents with metadata"""
    context_text = "\n\n".join([
    f"[Source: {doc.metadata.get('source', 'unknown')}] Page: {doc.metadata.get('page', 'unknown')}\n{doc.page_content.strip()}"
    for doc in docs])

    return context_text