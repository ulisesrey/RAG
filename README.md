# Retrieval-Augmented Generation (RAG)
This is an implementation of Retrieval-Augmented Generation (RAG) to answer questions from a very long document using Large Language Models (LLMs).

## Motivation
LLMs have context memory limits, meaning they predict the next word based on an input, but the input has to be short enough to fit in the memory. Otherwise the LLMs will only use the last words of the input. In practice, this means that we can't use LLMs to answer questions from very large sources.
Example of what an LLM can do:

input:
"Barcelona and Madrid have pollution levels way above the limits, on the other side Amsterdam has cleaner air.
Which cities should decrease their pollution levels?"


Example of what an LLM can not do:
input:
"Attached you find a report on the pollution levels of Madrid, Barcelona and Amsterdam. Each one is 100 pages. Which cities should decrease their pollution levels? "

## RAG Concept
Since LLMs can't have access to all the context, but we need the context to be able to answer the question, we can do the following:
1. Chunck our document into smaller parts.
2. Vectorize these parts in our documents via an embedding. Now every few sentences is a vector.
3. Vectorize the question (query) using the same embedding.
4. Retrieve the vectors closest to our queries. Since our texts are now vectors, we can calculate distance between vectors, and find the ones more similar to our question.
5. These vectors can be translated back to their text representation.
6. We can attach the text representation to our query to the LLMs.
7. Now the LLMs can answer the question based on our retrieved information, which fit the context limits of our LLM.

## Installation
Use the uv.lock and pyproject.toml files to install the environment.

## Generate the vector store
Put your desired pdf document in the code repository and specify the path in the config.yaml file.
Modify the other parameters in the config.yaml file according to your preferences.
Basically, choose an embedding model, define chunking parameters, and output paths for the vector store.

After that run the build_vector_store.py
```python
python build_vector_store.py
```
## Ask questions
Now you can ask questions, to do so run the retriever.py
```python
python retriever.py
```
Or if you want a more user friendly experience run the streamlit app with:
```python
streamlit run frontend/app.py
```
And follow the link, for example http://localhost:8501

You should see something like this:
![alt text](images/frontend_v1.png)

You can ask a question, and the answer will be generated using RAG.
In my example I used the book The origin of Species by Charles Darwin as a document to answer questions about the book.
![alt text](images/question.png)
After submitting the question it might take few seconds to provide the answer, since it needs to vectorize the question and retrieve the parts of the document that are more likely to contain the info.

And this is the answer:
![alt text](images/answer.png)
By clicking on Retrieved Source Chuncks you can see the part of the original document that was used to answer the question.

## Acknowledgments
This RAG implementation is based on https://github.com/emarco177/documentation-helper/blob/2-retrieval-qa/ingestion.py by Eden Marco.