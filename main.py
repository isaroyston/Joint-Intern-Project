from src.chatbot.withdrawal_chatbot import WithdrawalChatbot
from src.vector_store.vector_store import VectorStore
from dotenv import load_dotenv
import os

def main():
    load_dotenv()

    # Initialize vector store (must match ingest path + collection)
    vs = VectorStore(
        persist_directory="./vectordb",
        collection_name="sgbank_withdrawal_policy"
    )

    print(f"Documents in collection: {vs.get_collection_count()}")

    bot = WithdrawalChatbot(vector_store=vs)

    print("\nSGBank Withdrawal Assistant Ready.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        response = bot.chat(user_input)
        print("\nAssistant:", response, "\n")


if __name__ == "__main__":
    main()
