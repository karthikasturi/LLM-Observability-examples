"""
STAGE 1 — Basic Chatbot (No Observability Yet)

Purpose:
- Introduce LangChain + OpenAI without any observability
- Simple chatbot that accepts user input and generates responses
- Load API key from .env file
- No tracing or monitoring yet

What you'll learn:
- How to load API keys from .env file
- How to create a ChatOpenAI model instance
- How to build a simple interactive chatbot loop
- Basic prompt → response flow

Usage:
    python step1_basic_chatbot.py
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
# This keeps your API key secure and out of source code
load_dotenv()

def main():
    """
    Basic chatbot with no observability.
    Uses OpenAI's GPT model through LangChain.
    """
    
    print("=" * 70)
    print("STAGE 1: Basic Chatbot (No Observability)")
    print("=" * 70)
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY not found in .env file")
        print("Please create a .env file with your OpenAI API key")
        return
    
    print("\n✅ OpenAI API key loaded successfully")
    
    # Create a ChatOpenAI model instance
    # - model: specifies which OpenAI model to use (gpt-3.5-turbo is fast and cheap)
    # - temperature: controls randomness (0 = deterministic, 1 = creative)
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    print("✅ ChatOpenAI model initialized")
    print("\n🤖 Chatbot is ready! Type 'quit' or 'exit' to stop.\n")
    
    # Simple input loop for interactive chat
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye!")
            break
        
        # Skip empty inputs
        if not user_input:
            continue
        
        try:
            # Invoke the model with user input
            # This sends the request to OpenAI's API and waits for the response
            response = model.invoke(user_input)
            
            # Print the AI response
            # The response object contains the message content plus metadata
            print(f"\n🤖 Assistant: {response.content}\n")
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
    
    print("\n✅ Stage 1 Complete!")
    print("Next: Run step2_prompt_template.py to learn about prompt templates and chains")

if __name__ == "__main__":
    main()
