from src.core.llm import create_llm


def main():
    llm = create_llm()

    response = llm.invoke("Explain network slicing in one sentence.")

    print(response.content)


if __name__ == "__main__":
    main()
