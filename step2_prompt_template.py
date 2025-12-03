"""
STAGE 2 — Add PromptTemplate and Basic Chain

Purpose:
- Wrap the model using PromptTemplate for structured prompts
- Build a simple chain with a {user_message} variable
- Demonstrate chain.invoke() usage
- Show the benefits of prompt engineering

What you'll learn:
- How to create reusable prompt templates
- How to use variables in prompts with {placeholder} syntax
- How to build a chain: PromptTemplate → LLM → OutputParser
- How to invoke chains with different inputs

This is essential for building flexible, maintainable LLM applications.

Usage:
    python step2_prompt_template.py
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

def main():
    """
    Demonstrates prompt templates and chain construction.
    Shows how to build reusable prompts with variables.
    """
    
    print("=" * 70)
    print("STAGE 2: Prompt Templates and Chains")
    print("=" * 70)
    
    # Initialize the LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY not found in .env file")
        return
    
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    print("\n✅ ChatOpenAI model initialized")
    
    # Create a PromptTemplate
    # This allows you to create reusable prompts with variables
    # The {user_message} placeholder will be replaced with actual user input
    template = """You are a helpful AI assistant specializing in explaining technical concepts clearly and concisely.

User question: {user_message}

Please provide a clear, practical answer with examples if relevant."""
    
    prompt = PromptTemplate(
        input_variables=["user_message"],
        template=template
    )
    
    print("✅ PromptTemplate created with variable: {user_message}")
    
    # Create a chain: Prompt → LLM → Output Parser
    # This is the modern LangChain way using the pipe (|) operator
    # The chain will:
    # 1. Format the prompt with user input
    # 2. Send it to the LLM
    # 3. Parse the output as a string
    chain = prompt | model | StrOutputParser()
    
    print("✅ Chain built: PromptTemplate → LLM → OutputParser")
    
    # Test the chain with different inputs
    test_questions = [
        "What is LangChain and why is it useful?",
        "Explain the purpose of prompt templates.",
        "How do chains work in LangChain?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 70}")
        print(f"Example {i}:")
        print(f"📝 Question: {question}")
        print("=" * 70)
        
        # Invoke the chain with the user message
        # The chain automatically formats the prompt and gets a response
        response = chain.invoke({"user_message": question})
        
        print(f"\n🤖 Response:\n{response}\n")
    
    print("=" * 70)
    print("✅ Stage 2 Complete!")
    print("\nKey Takeaways:")
    print("- PromptTemplates make prompts reusable and maintainable")
    print("- Chains connect multiple components in a pipeline")
    print("- Variables in templates can be dynamically replaced")
    print("- The pipe operator (|) creates clean, readable chains")
    print("\nNext: Run step3_langsmith_tracing.py to enable LangSmith tracing")

if __name__ == "__main__":
    main()
