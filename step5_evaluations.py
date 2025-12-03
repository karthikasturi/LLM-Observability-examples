"""
STAGE 5 — Add Evaluation Logic (LLM-as-a-Judge)

Purpose:
- Define a simple evaluator function using @traceable
- Use ChatOpenAI to score responses on multiple dimensions
- Log evaluation results into LangSmith runs
- Print evaluation scores

What you'll learn:
- How to implement LLM-as-a-judge evaluation
- How to score responses on relevance, hallucination risk, and tone
- How LangSmith supports evaluation workflows
- How to structure evaluation prompts

This teaches how LangSmith supports LLM-based evaluations.

Usage:
    python step5_evaluations.py

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

# Load environment variables
load_dotenv()


@traceable(
    name="llm_evaluator",
    metadata={"evaluator_version": "1.0", "eval_type": "llm_as_judge"}
)
def evaluate_response(
    question: str,
    response: str,
    evaluator_model: ChatOpenAI
) -> dict:
    """
    LLM-as-a-judge evaluation function.
    
    This function uses an LLM to evaluate another LLM's response on:
    - Relevance: Does the response answer the question?
    - Hallucination Risk: Are there unsupported claims?
    - Tone: Is the tone appropriate and professional?
    
    Returns a dictionary with scores and reasoning.
    """
    
    # Evaluation prompt template
    eval_template = """You are an expert evaluator of AI-generated responses.

Evaluate the following response based on these criteria:

1. RELEVANCE (0-10): Does the response directly answer the question?
2. HALLUCINATION RISK (0-10): Are all claims supported and factual? (0=high risk, 10=no risk)
3. TONE (0-10): Is the tone appropriate, professional, and helpful?

Question: {question}

Response: {response}

Provide your evaluation in this exact JSON format:
{{
    "relevance_score": <0-10>,
    "relevance_reasoning": "<brief explanation>",
    "hallucination_score": <0-10>,
    "hallucination_reasoning": "<brief explanation>",
    "tone_score": <0-10>,
    "tone_reasoning": "<brief explanation>",
    "overall_score": <average of three scores>
}}

Provide ONLY the JSON, no additional text."""
    
    prompt = PromptTemplate(
        input_variables=["question", "response"],
        template=eval_template
    )
    
    # Create evaluation chain
    eval_chain = prompt | evaluator_model | StrOutputParser()
    
    # Get evaluation from LLM
    eval_result = eval_chain.invoke({
        "question": question,
        "response": response
    })
    
    # Parse JSON result
    try:
        # Try to extract JSON from the response
        eval_data = json.loads(eval_result.strip())
    except json.JSONDecodeError:
        # Fallback if LLM doesn't return valid JSON
        eval_data = {
            "relevance_score": 7,
            "relevance_reasoning": "Unable to parse evaluation",
            "hallucination_score": 7,
            "hallucination_reasoning": "Unable to parse evaluation",
            "tone_score": 7,
            "tone_reasoning": "Unable to parse evaluation",
            "overall_score": 7
        }
    
    return eval_data


@traceable(
    name="chat_and_evaluate",
    metadata={"workflow": "chat_with_evaluation"}
)
def chat_and_evaluate(
    user_message: str,
    chat_model: ChatOpenAI,
    evaluator_model: ChatOpenAI,
    chain
) -> dict:
    """
    Complete workflow: generate response and evaluate it.
    
    This creates a nested trace showing both the chat interaction
    and the evaluation, all visible in LangSmith.
    """
    
    # Generate response
    response = chain.invoke({"user_message": user_message})
    
    # Evaluate the response
    evaluation = evaluate_response(user_message, response, evaluator_model)
    
    return {
        "question": user_message,
        "response": response,
        "evaluation": evaluation
    }


def print_evaluation(result: dict):
    """Pretty print evaluation results."""
    eval_data = result["evaluation"]
    
    print(f"\n{'=' * 70}")
    print("📊 EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n📝 Question: {result['question']}")
    print(f"\n🤖 Response: {result['response']}")
    print(f"\n{'=' * 70}")
    print("SCORES:")
    print("=" * 70)
    
    print(f"\n🎯 Relevance: {eval_data['relevance_score']}/10")
    print(f"   Reasoning: {eval_data['relevance_reasoning']}")
    
    print(f"\n🔍 Hallucination Risk: {eval_data['hallucination_score']}/10")
    print(f"   Reasoning: {eval_data['hallucination_reasoning']}")
    
    print(f"\n💬 Tone: {eval_data['tone_score']}/10")
    print(f"   Reasoning: {eval_data['tone_reasoning']}")
    
    print(f"\n⭐ Overall Score: {eval_data['overall_score']:.1f}/10")
    print("=" * 70)


def main():
    """
    Demonstrates LLM-as-a-judge evaluation with LangSmith tracing.
    """
    
    print("=" * 70)
    print("STAGE 5: LLM-as-a-Judge Evaluation")
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
        print("Evaluation traces will not be logged to LangSmith.")
    else:
        print(f"\n✅ LangSmith API key loaded")
        print(f"✅ LangSmith project: {langsmith_project or 'default'}")
        print("\n📊 Evaluation traces will be logged to LangSmith")
    
    # Initialize models
    # Main chatbot model
    chat_model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Evaluator model (can use same or different model)
    evaluator_model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.0  # Use 0 for more consistent evaluations
    )
    
    print("✅ Chat model initialized (gpt-3.5-turbo, temp=0.7)")
    print("✅ Evaluator model initialized (gpt-3.5-turbo, temp=0.0)")
    
    # Create chatbot prompt
    template = """You are a helpful AI assistant specializing in technical topics.

User question: {user_message}

Please provide a clear, accurate, and helpful answer."""
    
    prompt = PromptTemplate(
        input_variables=["user_message"],
        template=template
    )
    
    # Build chain
    chain = prompt | chat_model | StrOutputParser()
    
    print("✅ Chat chain built")
    print("✅ Evaluation system ready")
    
    # Test questions
    test_questions = [
        "What is LangSmith and how does it help with LLM observability?",
        "Explain how LLM-as-a-judge evaluation works.",
    ]
    
    print("\n" + "=" * 70)
    print("Running chat + evaluation workflow...")
    print("=" * 70)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n\n{'#' * 70}")
        print(f"# Example {i}")
        print("#" * 70)
        
        # Run chat and evaluation
        result = chat_and_evaluate(
            user_message=question,
            chat_model=chat_model,
            evaluator_model=evaluator_model,
            chain=chain
        )
        
        # Print evaluation results
        print_evaluation(result)
    
    print("\n\n" + "=" * 70)
    print("✅ Stage 5 Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("- LLM-as-a-judge enables automated quality assessment")
    print("- Evaluation traces are nested within chat traces in LangSmith")
    print("- Multiple dimensions (relevance, hallucination, tone) provide comprehensive evaluation")
    print("- Evaluations can be logged and analyzed over time")
    
    print("\nEvaluation Best Practices:")
    print("- Use temperature=0 for evaluator model (more consistent)")
    print("- Provide clear scoring criteria in evaluation prompts")
    print("- Request structured outputs (JSON) for easier parsing")
    print("- Consider using stronger models (GPT-4) as evaluators")
    
    print("\nIn LangSmith, you can now:")
    print("- View evaluation traces alongside chat traces")
    print("- Compare evaluation scores across different runs")
    print("- Analyze trends in response quality")
    print("- Debug low-scoring responses")
    
    print("\nNext: Run step6_dataset_testing.py for dataset-based testing")

if __name__ == "__main__":
    main()
