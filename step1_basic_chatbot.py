"""
STAGE 1: Basic Chatbot
======================
This script demonstrates the most basic LangChain + OpenAI integration.

What you'll learn:
- How to load API keys from .env file
- How to create a ChatOpenAI model instance
- How to send a simple prompt and get a response
- Basic prompt → response flow

No observability yet - just the fundamentals.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
# This keeps your API key secure and out of source code
load_dotenv()

def main():
    print("=" * 60)
    print("STAGE 1: Basic Chatbot")
    print("=" * 60)
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env file")
        print("Please create a .env file with your OpenAI API key")
        return
    
    print("\n✓ API key loaded successfully")
    
    # Create a ChatOpenAI model instance
    # - model: specifies which OpenAI model to use (gpt-3.5-turbo is fast and cheap)
    # - temperature: controls randomness (0 = deterministic, 1 = creative)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=api_key
    )
    
    print("✓ ChatOpenAI model initialized")
    
    # Create a simple prompt
    prompt = "What is LangChain and why is observability important for LLM applications?"
    
    print(f"\n📝 Prompt: {prompt}")
    print("\n⏳ Waiting for response...\n")
    
    # Invoke the model with the prompt
    # This sends the request to OpenAI's API and waits for the response
    response = llm.invoke(prompt)
    
    # Print the response
    # The response object contains the message content plus metadata
    print("🤖 Response:")
    print("-" * 60)
    print(response.content)
    print("-" * 60)
    
    print("\n✅ Stage 1 Complete!")
    print("\nNext: Run step2_prompt_template.py to learn about prompt templates")

if __name__ == "__main__":
    main()
