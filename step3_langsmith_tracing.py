"""
STAGE 3 — Enable LangSmith Tracing

Purpose:
- Enable LangSmith by setting environment variables
- Use the @traceable decorator to trace function execution
- Demonstrate automatic tracing of LangChain chains
- Show how to view traces in LangSmith UI

What you'll learn:
- How to configure LangSmith with API key and project name
- How to use @traceable decorator for custom functions
- How LangChain automatically traces chains when LangSmith is enabled
- How to view and analyze traces in the LangSmith web UI

This introduces LangSmith's automatic tracing capabilities.

Usage:
    python step3_langsmith_tracing.py

Requirements:
    - LANGSMITH_API_KEY in .env
    - LANGSMITH_PROJECT in .env
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

# Load environment variables
load_dotenv()


@traceable
def chat_with_llm(user_message: str, model: ChatOpenAI, chain) -> str:
    """
    Traced chatbot function.
    
    The @traceable decorator automatically logs:
    - Function inputs (user_message)
    - Function outputs (response)
    - Execution time
    - Any errors that occur
    
    All of this appears in the LangSmith UI.
    """
    response = chain.invoke({"user_message": user_message})
    return response


def main():
    """
    Demonstrates LangSmith tracing with LangChain.
    Shows how to enable and use automatic tracing.
    """
    
    print("=" * 70)
    print("STAGE 3: Enable LangSmith Tracing")
    print("=" * 70)
    
    # Check for LangSmith configuration
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    langsmith_project = os.getenv("LANGSMITH_PROJECT")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("\n❌ ERROR: OPENAI_API_KEY not found in .env file")
        return
    
    if not langsmith_api_key:
        print("\n⚠️  WARNING: LANGSMITH_API_KEY not found in .env file")
        print("LangSmith tracing will not be enabled.")
        print("To enable tracing, add LANGSMITH_API_KEY to your .env file")
    else:
        print(f"\n✅ LangSmith API key loaded")
        print(f"✅ LangSmith project: {langsmith_project or 'default'}")
        print("\n📊 Tracing is ENABLED - all runs will be logged to LangSmith")
    
    # Initialize the LLM
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    print("✅ ChatOpenAI model initialized")
    
    # Create prompt template
    template = """You are a helpful AI assistant specializing in technical topics.

User question: {user_message}

Please provide a clear and concise answer."""
    
    prompt = PromptTemplate(
        input_variables=["user_message"],
        template=template
    )
    
    # Build chain
    # When LangSmith is enabled, chains are automatically traced
    chain = prompt | model | StrOutputParser()
    
    print("✅ Chain built (automatically traced when LangSmith is enabled)")
    
    # Test questions
    test_questions = [
        "What is LangSmith?",
        "Why is tracing important for LLM applications?"
    ]
    
    print("\n" + "=" * 70)
    print("Running traced examples...")
    print("=" * 70)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Question {i}: {question}")
        
        # Call the traced function
        # This will create a trace in LangSmith
        response = chat_with_llm(question, model, chain)
        
        print(f"🤖 Response: {response}\n")
    
    print("=" * 70)
    print("✅ Stage 3 Complete!")
    print("\nKey Takeaways:")
    print("- LangSmith automatically traces LangChain operations")
    print("- @traceable decorator adds custom function tracing")
    print("- Traces include inputs, outputs, latency, and errors")
    print("- View traces in LangSmith UI: https://smith.langchain.com/")
    print("\nTo view your traces:")
    print("1. Go to https://smith.langchain.com/")
    print("2. Select your project")
    print("3. View the trace details for each run")
    print("\nNext: Run step4_tracing_metadata.py to add custom metadata")

if __name__ == "__main__":
    main()
