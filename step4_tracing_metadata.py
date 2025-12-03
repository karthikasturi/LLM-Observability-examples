"""
STAGE 4 — Add Rich Tracing with Custom Metadata

Purpose:
- Add custom run metadata to traces
- Add custom tags for run categorization
- Show how these appear in LangSmith
- Demonstrate nested traced functions using @traceable

What you'll learn:
- How to add custom metadata to LangSmith runs
- How to use tags for organizing and filtering traces
- How to create nested traces for complex workflows
- How structured observability improves debugging

This shows structured observability and run categorization.

Usage:
    python step4_tracing_metadata.py

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


@traceable(
    name="preprocess_input",
    metadata={"stage": "preprocessing", "module": "chatbot"}
)
def preprocess_input(user_message: str) -> str:
    """
    Preprocess user input before sending to LLM.
    
    This is a nested traced function - it will appear as a child span
    in the LangSmith trace hierarchy.
    
    The @traceable decorator automatically logs:
    - Input: user_message
    - Output: cleaned message
    - Execution time
    """
    # Simple preprocessing: strip whitespace and normalize
    cleaned = user_message.strip()
    print(f"   🔧 Preprocessed: '{user_message}' -> '{cleaned}'")
    return cleaned


@traceable(
    name="chatbot_interaction",
    metadata={
        "environment": "local",
        "module": "chatbot-training",
        "version": "1.0"
    },
    tags=["chat", "demo", "user-session"]
)
def chat_with_metadata(user_message: str, model: ChatOpenAI, chain) -> str:
    """
    Main chatbot function with rich metadata and tags.
    
    The metadata will appear in LangSmith and help you:
    - Filter runs by environment (local, staging, prod)
    - Identify which module/version produced the run
    - Track user sessions
    
    Tags help categorize and search for specific types of runs.
    """
    # Call nested traced function
    # This will create a nested span in LangSmith
    processed_input = preprocess_input(user_message)
    
    # Invoke the chain
    response = chain.invoke({"user_message": processed_input})
    
    return response


def main():
    """
    Demonstrates LangSmith tracing with custom metadata and tags.
    Shows how to structure observability for complex workflows.
    """
    
    print("=" * 70)
    print("STAGE 4: Rich Tracing with Custom Metadata and Tags")
    print("=" * 70)
    
    # Check for configuration
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    langsmith_project = os.getenv("LANGSMITH_PROJECT")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("\n❌ ERROR: OPENAI_API_KEY not found in .env file")
        return
    
    if not langsmith_api_key:
        print("\n⚠️  WARNING: LANGSMITH_API_KEY not found in .env file")
        print("LangSmith tracing will not be enabled.")
    else:
        print(f"\n✅ LangSmith API key loaded")
        print(f"✅ LangSmith project: {langsmith_project or 'default'}")
        print("\n📊 Tracing with METADATA and TAGS enabled")
    
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
    chain = prompt | model | StrOutputParser()
    
    print("✅ Chain built")
    
    # Demonstrate metadata and tags
    print("\n" + "=" * 70)
    print("Metadata added to traces:")
    print("=" * 70)
    print("  • environment: local")
    print("  • module: chatbot-training")
    print("  • version: 1.0")
    print("\nTags added to traces:")
    print("  • chat")
    print("  • demo")
    print("  • user-session")
    print("=" * 70)
    
    # Test questions with metadata
    test_questions = [
        "What is metadata in the context of observability?",
        "Why are tags useful for organizing traces?",
        "How do nested traces help with debugging?"
    ]
    
    print("\n" + "=" * 70)
    print("Running traced examples with metadata...")
    print("=" * 70)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Question {i}: {question}")
        
        # Call the traced function with metadata
        # This will create a rich trace in LangSmith with:
        # - Custom metadata (environment, module, version)
        # - Custom tags (chat, demo, user-session)
        # - Nested spans (preprocess_input -> chain invocation)
        response = chat_with_metadata(question, model, chain)
        
        print(f"🤖 Response: {response}\n")
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("✅ Stage 4 Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("- Metadata adds structured context to traces")
    print("- Tags enable easy filtering and categorization")
    print("- Nested @traceable functions create hierarchical traces")
    print("- Rich observability makes debugging much easier")
    
    print("\nIn LangSmith UI, you can now:")
    print("1. Filter runs by metadata (environment=local)")
    print("2. Search runs by tags (tag:demo)")
    print("3. View nested trace hierarchy")
    print("4. Compare runs across different environments")
    
    print("\nNext: Run step5_evaluations.py to add LLM-as-a-judge evaluation")

if __name__ == "__main__":
    main()
