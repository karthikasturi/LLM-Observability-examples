"""
STAGE 2: Add LangChain PromptTemplate and Chain
===============================================
This script demonstrates how to use PromptTemplate for prompt engineering.

What you'll learn:
- How to create reusable prompt templates
- How to use variables in prompts
- How to build a simple chain with prompt + LLM
- How to run the chain with different inputs

This is essential for building flexible, maintainable LLM applications.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

def main():
    print("=" * 60)
    print("STAGE 2: Prompt Templates and Chains")
    print("=" * 60)
    
    # Initialize the LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env file")
        return
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=api_key
    )
    
    print("\n✓ ChatOpenAI model initialized")
    
    # Create a PromptTemplate
    # This allows you to create reusable prompts with variables
    # The {user_input} placeholder will be replaced with actual values
    template = """You are a helpful AI assistant specializing in explaining technical concepts.
    
User question: {user_input}

Please provide a clear, concise answer with practical examples."""
    
    prompt = PromptTemplate(
        input_variables=["user_input"],
        template=template
    )
    
    print("✓ PromptTemplate created")
    
    # Create a chain: Prompt → LLM → Output Parser
    # This is the modern LangChain way using the pipe operator
    # The chain will:
    # 1. Format the prompt with user input
    # 2. Send it to the LLM
    # 3. Parse the output as a string
    chain = prompt | llm | StrOutputParser()
    
    print("✓ Chain built: PromptTemplate → LLM → OutputParser")
    
    # Test the chain with different inputs
    test_questions = [
        "What are the benefits of using prompt templates in LangChain?",
        "How does token counting work in OpenAI models?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 60}")
        print(f"Question {i}:")
        print(f"📝 {question}")
        print(f"{'=' * 60}")
        print("\n⏳ Processing...\n")
        
        # Run the chain with the user input
        # The chain automatically formats the prompt and invokes the LLM
        response = chain.invoke({"user_input": question})
        
        print("🤖 Response:")
        print("-" * 60)
        print(response)
        print("-" * 60)
    
    print("\n✅ Stage 2 Complete!")
    print("\nKey Takeaways:")
    print("- PromptTemplates make prompts reusable and maintainable")
    print("- Chains connect multiple components in a pipeline")
    print("- Variables in templates can be dynamically replaced")
    print("\nNext: Run step3_basic_callback.py to add observability hooks")

if __name__ == "__main__":
    main()
