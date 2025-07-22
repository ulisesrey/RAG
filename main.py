from langchain_ollama import ChatOllama



def main():
    llm = ChatOllama(model="mistral:latest")
    result = llm.invoke("What is the capital of France?")
    print(result.content)

if __name__ == "__main__":
    main()
