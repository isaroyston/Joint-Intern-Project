"""
Example usage of the chatbot framework
Demonstrates basic chatbot, vector store, and prompt attack testing
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from vector_store import VectorStore
from chatbot import Chatbot
from datasets import PromptAttackDataset


def example_vector_store():
    """Example: Using the vector store"""
    print("=== Vector Store Example ===\n")
    
    # Initialize vector store
    vector_store = VectorStore(
        persist_directory="./data/example_db",
        collection_name="example_collection"
    )
    
    # Add sample documents
    documents = [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "Vector databases are used for similarity search.",
        "Chatbots can use RAG (Retrieval-Augmented Generation) for better responses."
    ]
    
    vector_store.add_documents(documents)
    print(f"Added {len(documents)} documents to vector store")
    print(f"Total documents in collection: {vector_store.get_collection_count()}\n")
    
    # Search for similar documents
    query = "What is Python?"
    results = vector_store.search(query, n_results=2)
    
    print(f"Query: {query}")
    print("Top results:")
    for i, doc in enumerate(results['documents'][0]):
        print(f"  {i+1}. {doc}")
    print()


def example_chatbot_without_api():
    """Example: Using chatbot without OpenAI API (vector store only)"""
    print("=== Chatbot Example (No API) ===\n")
    
    # Initialize vector store with knowledge
    vector_store = VectorStore(
        persist_directory="./data/chatbot_db",
        collection_name="knowledge_base"
    )
    
    knowledge = [
        "Our company offers 24/7 customer support via email and phone.",
        "We ship products worldwide with free shipping on orders over $50.",
        "Our return policy allows returns within 30 days of purchase.",
        "We accept payments via credit card, PayPal, and bank transfer."
    ]
    
    vector_store.add_documents(knowledge)
    
    # Initialize chatbot (will work without API key for demo mode)
    try:
        chatbot = Chatbot(
            api_key="dummy_key",  # Not used in demo mode
            vector_store=vector_store
        )
    except:
        # If OpenAI client fails, we'll just demonstrate the vector store
        print("Note: Using demo mode without OpenAI API\n")
    
    # Query the knowledge base
    query = "What is your return policy?"
    print(f"User: {query}")
    
    # Use vector store directly for demo
    results = vector_store.search(query, n_results=1)
    if results['documents'][0]:
        print(f"Assistant: {results['documents'][0][0]}")
    print()


def example_prompt_attacks():
    """Example: Using prompt attack dataset"""
    print("=== Prompt Attack Dataset Example ===\n")
    
    # Initialize dataset
    dataset = PromptAttackDataset()
    
    print(f"Total attack patterns: {len(dataset.get_all_attacks())}")
    print(f"Categories: {', '.join(dataset.get_categories())}\n")
    
    # Show high severity attacks
    high_severity = dataset.get_attacks_by_severity("high")
    print(f"High severity attacks: {len(high_severity)}")
    for attack in high_severity[:3]:
        print(f"  - {attack['name']}: {attack['description']}")
    print()
    
    # Show attacks by category
    injection_attacks = dataset.get_attacks_by_category("instruction_override")
    print(f"Instruction override attacks: {len(injection_attacks)}")
    for attack in injection_attacks:
        print(f"  - {attack['name']}")
    print()


def main():
    """Run all examples"""
    print("=" * 60)
    print("Chatbot Framework Examples")
    print("=" * 60)
    print()
    
    try:
        example_vector_store()
    except Exception as e:
        print(f"Vector store example error: {e}\n")
    
    try:
        example_chatbot_without_api()
    except Exception as e:
        print(f"Chatbot example error: {e}\n")
    
    try:
        example_prompt_attacks()
    except Exception as e:
        print(f"Prompt attacks example error: {e}\n")
    
    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
