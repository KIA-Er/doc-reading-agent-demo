from src.code.rag_workflow.rag import Retriever
import asyncio

def main():
    retriever = Retriever()
    while True:
        query = input("请输入您的问题：")
        if query == "exit":
            break
        response = asyncio.run(retriever.retieve(query=query))
        print(response)


if __name__ == "__main__":
    main()