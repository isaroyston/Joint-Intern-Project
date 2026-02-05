"""
Chatbot Module
Provides a conversational AI interface with RAG capabilities using vector store
"""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_store.vector_store import VectorStore


class Chatbot:
    """Chatbot with Retrieval-Augmented Generation (RAG) capabilities"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 temperature: float = 0.7,
                 max_tokens: int = 500,
                 vector_store: Optional[VectorStore] = None):
        """
        Initialize the chatbot
        
        Args:
            api_key: OpenAI API key (if None, loads from environment)
            model: OpenAI model name
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
            vector_store: Optional VectorStore instance for RAG
        """
        # Load environment variables
        load_dotenv()
        
        # Set API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.vector_store = vector_store
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        # System prompt
        self.system_prompt = """You are a helpful AI assistant. You provide accurate, 
        helpful, and friendly responses to user queries. If you're provided with context 
        from a knowledge base, use it to inform your responses."""
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set custom system prompt"""
        self.system_prompt = prompt
    
    def add_to_history(self, role: str, content: str) -> None:
        """Add message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
    
    def _retrieve_context(self, query: str, n_results: int = 3) -> str:
        """Retrieve relevant context from vector store"""
        if not self.vector_store:
            return ""
        
        results = self.vector_store.search(query, n_results=n_results)
        
        if not results['documents'] or not results['documents'][0]:
            return ""
        
        context_docs = results['documents'][0]
        context = "\n\n".join([f"Context {i+1}: {doc}" for i, doc in enumerate(context_docs)])
        return context
    
    def chat(self, 
             user_message: str, 
             use_rag: bool = True,
             stream: bool = False) -> str:
        """
        Send a message and get a response
        
        Args:
            user_message: User's message
            use_rag: Whether to use RAG for enhanced responses
            stream: Whether to stream the response (not implemented)
            
        Returns:
            Assistant's response
        """
        # Retrieve context if RAG is enabled
        context = ""
        if use_rag and self.vector_store:
            context = self._retrieve_context(user_message)
        
        # Build the message with context
        if context:
            enhanced_message = f"Relevant Context:\n{context}\n\nUser Question: {user_message}"
        else:
            enhanced_message = user_message
        
        # Add to history
        self.add_to_history("user", user_message)
        
        # Build messages for API
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history (last 10 messages)
        messages.extend(self.conversation_history[-10:])
        
        # Update last user message with context
        if context:
            messages[-1]["content"] = enhanced_message
        
        # Call OpenAI API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            assistant_message = response.choices[0].message.content
            
            # Add to history
            self.add_to_history("assistant", assistant_message)
            
            return assistant_message
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            return error_msg
    
    def chat_without_api(self, user_message: str) -> str:
        """
        Chat without calling OpenAI API (for testing/demo)
        Uses only the vector store for responses
        
        Args:
            user_message: User's message
            
        Returns:
            Response based on vector store results
        """
        if not self.vector_store:
            return "No knowledge base available. Please add documents to the vector store."
        
        context = self._retrieve_context(user_message, n_results=2)
        
        if not context:
            return "I couldn't find relevant information in my knowledge base."
        
        return f"Based on my knowledge:\n\n{context}"
