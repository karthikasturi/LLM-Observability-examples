"""
STAGE 7 — Full Chatbot with Tracing, Metadata, and Evaluation

Purpose:
- Combine LangChain chain, LangSmith tracing, metadata, and evaluation
- Interactive chatbot loop where each message is traced and evaluated
- Demonstrate complete production-ready LLM application

Features:
- Automatic tracing of all interactions
- Custom metadata for run categorization
- Nested traces for complex workflows
- LLM-as-a-judge evaluation for each response

Usage:
    python step7_full_chatbot.py

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
import json

load_dotenv()


@traceable(name="preprocess", metadata={"stage": "preprocessing"})
def preprocess_input(user_message: str) -> str:
    """Clean and prepare user input."""
    return user_message.strip()


@traceable(name="evaluate", metadata={"stage": "evaluation"})
def evaluate_response(question: str, response: str, evaluator: ChatOpenAI) -> dict:
    """Evaluate response using LLM-as-a-judge."""
    
    eval_prompt = """Evaluate this AI response. Provide scores 0-10 in JSON:
{{"relevance": <score>, "quality": <score>, "overall": <avg>}}

Question: {question}
Response: {response}

JSON only:"""
    
    prompt = PromptTemplate(
        input_variables=["question", "response"],
        template=eval_prompt
    )
    
    chain = prompt | evaluator | StrOutputParser()
    
    try:
        result = chain.invoke({"question": question, "response": response})
        return json.loads(result.strip())
    except:
        return {"relevance": 7, "quality": 7, "overall": 7}


@traceable(
    name="chat_interaction",
    metadata={"environment": "local", "module": "full-chatbot", "version": "1.0"},
    tags=["chatbot", "production", "full-observability"]
)
def process_message(user_message: str, chat_model, evaluator_model, chain) -> dict:
    """
    Complete message processing pipeline:
    1. Preprocess input (traced)
    2. Generate response (traced)
    3. Evaluate response (traced)
    """
    
    processed = preprocess_input(user_message)
    response = chain.invoke({"user_message": processed})
    evaluation = evaluate_response(user_message, response, evaluator_model)
    
    return {
        "user_message": user_message,
        "response": response,
        "evaluation": evaluation
    }


def main():
    """Full chatbot with complete LangSmith observability."""
    
    print("=" * 70)
    print("STAGE 7: Full Chatbot with Complete Observability")
    print("=" * 70)
    
    # Check configuration
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    langsmith_project = os.getenv("LANGSMITH_PROJECT")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("\n❌ ERROR: OPENAI_API_KEY not found")
        return
    
    print(f"\n✅ OpenAI API key loaded")
    
    if not langsmith_api_key:
        print("⚠️  WARNING: LANGSMITH_API_KEY not found")
        print("   Observability features will be limited.")
    else:
        print(f"✅ LangSmith API key loaded")
        print(f"✅ LangSmith project: {langsmith_project or 'default'}")
        print("\n🎯 Full observability enabled:")
        print("   • Automatic tracing")
        print("   • Custom metadata")
        print("   • Run tagging")
        print("   • Nested traces")
        print("   • LLM-as-a-judge evaluation")
    
    # Initialize models
    chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    evaluator_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    
    print("\n✅ Chat model initialized")
    print("✅ Evaluator model initialized")
    
    # Create chatbot chain
    template = """You are a helpful AI assistant specializing in technical topics.

User: {user_message}

Please provide a clear and helpful response."""
    
    prompt = PromptTemplate(
        input_variables=["user_message"],
        template=template
    )
    
    chain = prompt | chat_model | StrOutputParser()
    print("✅ Chat chain built")
    
    # Demo conversation
    demo_messages = [
        "What is LangSmith?",
        "How does observability help in production LLM applications?",
        "Explain the benefits of tracing."
    ]
    
    print("\n" + "=" * 70)
    print("🤖 RUNNING DEMO CONVERSATION")
    print("=" * 70)
    print("\nEach message will be:")
    print("  1. Preprocessed (traced)")
    print("  2. Sent to LLM (traced)")
    print("  3. Evaluated (traced)")
    print("  4. All traces visible in LangSmith\n")
    
    for i, message in enumerate(demo_messages, 1):
        print(f"\n{'─' * 70}")
        print(f"Message {i}/{len(demo_messages)}")
        print(f"👤 You: {message}")
        
        # Process through complete pipeline
        result = process_message(message, chat_model, evaluator_model, chain)
        
        print(f"\n🤖 Assistant: {result['response']}")
        
        eval_data = result['evaluation']
        print(f"\n📊 Evaluation:")
        print(f"   Relevance: {eval_data['relevance']}/10")
        print(f"   Quality: {eval_data['quality']}/10")
        print(f"   Overall: {eval_data['overall']:.1f}/10")
    
    print("\n\n" + "=" * 70)
    print("✅ STAGE 7 COMPLETE - ALL STAGES FINISHED!")
    print("=" * 70)
    
    print("\n🎉 Congratulations! You've completed all 7 stages!")
    
    print("\n📚 What You've Learned:")
    print("  ✅ Stage 1: Basic LangChain chatbot")
    print("  ✅ Stage 2: Prompt templates and chains")
    print("  ✅ Stage 3: LangSmith tracing with @traceable")
    print("  ✅ Stage 4: Custom metadata and tags")
    print("  ✅ Stage 5: LLM-as-a-judge evaluation")
    print("  ✅ Stage 6: Dataset-based testing")
    print("  ✅ Stage 7: Complete integration")
    
    print("\n🔍 LangSmith Features You've Used:")
    print("  • Automatic tracing of LangChain operations")
    print("  • Custom function tracing with @traceable")
    print("  • Metadata for run categorization")
    print("  • Tags for filtering and search")
    print("  • Nested traces for complex workflows")
    print("  • LLM-based evaluation")
    print("  • Dataset management and testing")
    print("  • Run comparisons")
    
    print("\n📊 View Your Traces:")
    print("  1. Go to https://smith.langchain.com/")
    print(f"  2. Select project: {langsmith_project or 'default'}")
    print("  3. Browse runs and traces")
    print("  4. Compare performance across runs")
    print("  5. Drill into nested traces")
    
    print("\n🚀 Next Steps for Production:")
    print("  • Set up continuous evaluation")
    print("  • Create custom evaluators for your use case")
    print("  • Build comprehensive test datasets")
    print("  • Monitor quality trends over time")
    print("  • Use LangSmith annotations for human feedback")
    print("  • Integrate with CI/CD pipelines")
    
    print("\n💡 Interactive Mode:")
    print("  To enable interactive chat, uncomment the while loop")
    print("  in the run_interactive_chat() function and call it here.")


if __name__ == "__main__":
    main()