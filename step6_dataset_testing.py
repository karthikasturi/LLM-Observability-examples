"""
STAGE 6 — Add Dataset-Based Testing in LangSmith

Purpose:
- Create a small in-code dataset with test cases
- Convert it into a LangSmith dataset
- Run batch evaluations using client.run_on_dataset()
- Evaluate correctness and consistency
- Print summary results

What you'll learn:
- How to create and manage datasets in LangSmith
- How to run batch evaluations on datasets
- How to compare expected vs actual outputs
- How dataset-driven testing prevents regressions

This demonstrates dataset-driven regression testing.

Usage:
    python step6_dataset_testing.py

Requirements:
    - LANGSMITH_API_KEY in .env
    - LANGSMITH_PROJECT in .env
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

# Load environment variables
load_dotenv()


# Define test dataset
TEST_DATASET = [
    {
        "input": "What is Kubernetes?",
        "expected": "Kubernetes is an open-source container orchestration platform that automates deployment, scaling, and management of containerized applications."
    },
    {
        "input": "Explain Terraform.",
        "expected": "Terraform is an infrastructure-as-code tool that allows you to define and provision infrastructure using declarative configuration files."
    },
    {
        "input": "Define SRE.",
        "expected": "SRE (Site Reliability Engineering) is a discipline that applies software engineering principles to operations tasks, focusing on reliability, scalability, and automation."
    },
    {
        "input": "What is observability?",
        "expected": "Observability is the ability to understand the internal state of a system by examining its outputs (logs, metrics, and traces)."
    }
]


@traceable
def run_chatbot(input_text: str, chain) -> str:
    """
    Run the chatbot chain on a single input.
    This will be traced for each dataset example.
    """
    return chain.invoke({"user_message": input_text})


def create_dataset(client: Client, dataset_name: str) -> str:
    """
    Create a LangSmith dataset from our test cases.
    
    Returns the dataset name.
    """
    print(f"\n📊 Creating dataset: '{dataset_name}'")
    
    # Check if dataset already exists
    try:
        existing_datasets = list(client.list_datasets())
        for ds in existing_datasets:
            if ds.name == dataset_name:
                print(f"✅ Dataset '{dataset_name}' already exists")
                return dataset_name
    except Exception as e:
        print(f"⚠️  Could not check existing datasets: {e}")
    
    # Create the dataset
    try:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Test dataset for chatbot evaluation"
        )
        print(f"✅ Created dataset with ID: {dataset.id}")
        
        # Add examples to the dataset
        for i, example in enumerate(TEST_DATASET, 1):
            client.create_example(
                dataset_id=dataset.id,
                inputs={"input": example["input"]},
                outputs={"expected": example["expected"]}
            )
            print(f"   ✅ Added example {i}/{len(TEST_DATASET)}")
        
        print(f"✅ Dataset '{dataset_name}' created with {len(TEST_DATASET)} examples")
        return dataset_name
    
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return None


def evaluate_result(run_outputs: dict, example_outputs: dict) -> dict:
    """
    Simple evaluator function that checks if the response is relevant.
    
    In production, you'd use more sophisticated evaluation logic,
    possibly with LLM-as-a-judge or semantic similarity.
    """
    actual = run_outputs.get("output", "")
    expected = example_outputs.get("expected", "")
    
    # Simple heuristic: check if response is non-empty and reasonably long
    is_valid = len(actual) > 20
    
    # Check if any key terms from expected answer appear in actual
    expected_words = set(expected.lower().split())
    actual_words = set(actual.lower().split())
    overlap = len(expected_words.intersection(actual_words))
    
    relevance_score = min(1.0, overlap / max(len(expected_words) * 0.3, 1))
    
    return {
        "key": "correctness",
        "score": relevance_score,
        "comment": f"Valid response: {is_valid}, Relevance: {relevance_score:.2f}"
    }


def run_dataset_evaluation(
    client: Client,
    dataset_name: str,
    chain
) -> dict:
    """
    Run evaluation on the entire dataset.
    
    This uses LangSmith's run_on_dataset functionality to:
    - Run the chain on each dataset example
    - Collect results
    - Apply evaluators
    - Generate summary statistics
    """
    print(f"\n🔬 Running evaluation on dataset: '{dataset_name}'")
    print("=" * 70)
    
    results = []
    
    # Get dataset
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return None
    
    # Get examples
    examples = list(client.list_examples(dataset_id=dataset.id))
    print(f"📝 Found {len(examples)} examples in dataset")
    
    # Run evaluation on each example
    for i, example in enumerate(examples, 1):
        input_text = example.inputs.get("input", "")
        expected = example.outputs.get("expected", "")
        
        print(f"\n{'─' * 70}")
        print(f"Example {i}/{len(examples)}")
        print(f"📝 Input: {input_text}")
        print(f"✓ Expected: {expected[:100]}...")
        
        try:
            # Run the chatbot
            actual = run_chatbot(input_text, chain)
            print(f"🤖 Actual: {actual[:100]}...")
            
            # Evaluate
            eval_result = evaluate_result(
                {"output": actual},
                {"expected": expected}
            )
            
            print(f"📊 Score: {eval_result['score']:.2f}")
            print(f"💬 Comment: {eval_result['comment']}")
            
            results.append({
                "input": input_text,
                "expected": expected,
                "actual": actual,
                "score": eval_result["score"]
            })
            
        except Exception as e:
            print(f"❌ Error running example: {e}")
            results.append({
                "input": input_text,
                "expected": expected,
                "actual": None,
                "score": 0.0,
                "error": str(e)
            })
    
    return results


def print_summary(results: list):
    """Print summary statistics of the evaluation."""
    print("\n\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY")
    print("=" * 70)
    
    if not results:
        print("❌ No results to summarize")
        return
    
    total = len(results)
    scores = [r["score"] for r in results if "score" in r]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    passed = sum(1 for s in scores if s >= 0.5)
    failed = total - passed
    
    print(f"\n📈 Total Examples: {total}")
    print(f"✅ Passed (score >= 0.5): {passed}")
    print(f"❌ Failed (score < 0.5): {failed}")
    print(f"⭐ Average Score: {avg_score:.2f}")
    
    print("\n" + "=" * 70)
    print("DETAILED RESULTS:")
    print("=" * 70)
    
    for i, result in enumerate(results, 1):
        status = "✅" if result.get("score", 0) >= 0.5 else "❌"
        print(f"\n{status} Example {i}: {result['input']}")
        print(f"   Score: {result.get('score', 0):.2f}")


def main():
    """
    Demonstrates dataset-based testing with LangSmith.
    """
    
    print("=" * 70)
    print("STAGE 6: Dataset-Based Testing in LangSmith")
    print("=" * 70)
    
    # Check for configuration
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    langsmith_project = os.getenv("LANGSMITH_PROJECT")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("\n❌ ERROR: OPENAI_API_KEY not found in .env file")
        return
    
    if not langsmith_api_key:
        print("\n❌ ERROR: LANGSMITH_API_KEY not found in .env file")
        print("Dataset testing requires LangSmith.")
        print("Please add LANGSMITH_API_KEY to your .env file")
        return
    
    print(f"\n✅ LangSmith API key loaded")
    print(f"✅ LangSmith project: {langsmith_project or 'default'}")
    
    # Initialize LangSmith client
    client = Client()
    print("✅ LangSmith client initialized")
    
    # Initialize chat model
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    print("✅ ChatOpenAI model initialized")
    
    # Create prompt template
    template = """You are a helpful AI assistant that explains technical concepts clearly.

User question: {user_message}

Please provide a clear and accurate explanation."""
    
    prompt = PromptTemplate(
        input_variables=["user_message"],
        template=template
    )
    
    # Build chain
    chain = prompt | model | StrOutputParser()
    print("✅ Chat chain built")
    
    # Create dataset
    dataset_name = "technical-qa-dataset"
    create_dataset(client, dataset_name)
    
    # Run evaluation
    results = run_dataset_evaluation(client, dataset_name, chain)
    
    # Print summary
    if results:
        print_summary(results)
    
    print("\n\n" + "=" * 70)
    print("✅ Stage 6 Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("- Datasets enable systematic testing of LLM applications")
    print("- Batch evaluation helps catch regressions")
    print("- LangSmith tracks evaluation results over time")
    print("- You can compare runs and track improvements")
    
    print("\nDataset Best Practices:")
    print("- Include diverse test cases covering edge cases")
    print("- Update datasets as you discover new failure modes")
    print("- Run evaluations on every major change")
    print("- Use datasets for A/B testing different prompts")
    
    print("\nIn LangSmith, you can now:")
    print("- View all dataset runs in the UI")
    print("- Compare results across different model versions")
    print("- Drill down into individual failures")
    print("- Track performance trends over time")
    
    print("\nNext: Run step7_full_chatbot.py for complete integration")

if __name__ == "__main__":
    main()
