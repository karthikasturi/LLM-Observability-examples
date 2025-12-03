"""
STAGE 4: Cost Tracking
======================
This script adds cost estimation to observability metrics.

What you'll learn:
- How to track prompt and completion tokens separately
- How to calculate estimated costs using OpenAI pricing
- How to accumulate costs across multiple requests
- Why cost tracking is critical for production LLM apps

OpenAI Pricing (as of reference, check current pricing):
- GPT-3.5-turbo: $0.0015 per 1K prompt tokens, $0.002 per 1K completion tokens
- GPT-4: $0.03 per 1K prompt tokens, $0.06 per 1K completion tokens
"""

import os
import time
from datetime import datetime
from typing import Any, Dict, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler

# Load environment variables
load_dotenv()


class CostTrackingCallback(BaseCallbackHandler):
    """
    Enhanced callback handler with cost tracking capabilities.
    
    This handler extends basic observability to include:
    - Token usage tracking (prompt vs completion)
    - Cost estimation based on OpenAI pricing
    - Cumulative cost tracking across requests
    """
    
    # OpenAI Pricing (per 1K tokens)
    # Update these values based on current OpenAI pricing
    PRICING = {
        "gpt-3.5-turbo": {
            "prompt": 0.0015,      # $0.0015 per 1K prompt tokens
            "completion": 0.002     # $0.002 per 1K completion tokens
        },
        "gpt-4": {
            "prompt": 0.03,         # $0.03 per 1K prompt tokens
            "completion": 0.06      # $0.06 per 1K completion tokens
        },
        "gpt-4-turbo": {
            "prompt": 0.01,         # $0.01 per 1K prompt tokens
            "completion": 0.03      # $0.03 per 1K completion tokens
        }
    }
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        super().__init__()
        self.model_name = model_name
        self.start_time = None
        self.end_time = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.request_cost = 0.0
        self.cumulative_cost = 0.0
        self.request_count = 0
    
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate the cost of a single request.
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
        
        Returns:
            Total cost in USD
        """
        # Get pricing for the model
        pricing = self.PRICING.get(self.model_name, self.PRICING["gpt-3.5-turbo"])
        
        # Calculate costs (divide by 1000 because pricing is per 1K tokens)
        prompt_cost = (prompt_tokens / 1000.0) * pricing["prompt"]
        completion_cost = (completion_tokens / 1000.0) * pricing["completion"]
        
        return prompt_cost + completion_cost
    
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> None:
        """Called when LLM starts processing."""
        self.start_time = time.time()
        self.request_count += 1
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n" + "=" * 60)
        print(f"🕐 Request #{self.request_count} | Start Time: {timestamp}")
        print("=" * 60)
        print("\n📤 Prompt:")
        print("-" * 60)
        for prompt in prompts:
            print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        print("-" * 60)
    
    def on_llm_end(
        self, 
        response: Any, 
        **kwargs: Any
    ) -> None:
        """Called when LLM finishes processing."""
        self.end_time = time.time()
        latency = self.end_time - self.start_time
        
        # Extract token usage
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            self.prompt_tokens = token_usage.get('prompt_tokens', 0)
            self.completion_tokens = token_usage.get('completion_tokens', 0)
            self.total_tokens = token_usage.get('total_tokens', 0)
        
        # Calculate cost for this request
        self.request_cost = self.calculate_cost(self.prompt_tokens, self.completion_tokens)
        self.cumulative_cost += self.request_cost
        
        # Print observability metrics with cost
        print("\n📊 Observability Metrics:")
        print("-" * 60)
        print(f"⏱️  Latency: {latency:.3f} seconds")
        print(f"🎯 Prompt Tokens: {self.prompt_tokens}")
        print(f"💬 Completion Tokens: {self.completion_tokens}")
        print(f"📈 Total Tokens: {self.total_tokens}")
        print("-" * 60)
        
        # Print cost metrics
        print("\n💰 Cost Analysis:")
        print("-" * 60)
        print(f"💵 This Request Cost: ${self.request_cost:.6f}")
        print(f"📊 Cumulative Cost: ${self.cumulative_cost:.6f}")
        print(f"📈 Average Cost/Request: ${self.cumulative_cost / self.request_count:.6f}")
        print(f"🔢 Total Requests: {self.request_count}")
        print("-" * 60)
        
        # Estimate monthly cost projections
        if self.request_count >= 1:
            daily_cost = self.cumulative_cost * (1000 / self.request_count)  # Assuming 1000 requests/day
            monthly_cost = daily_cost * 30
            print("\n📈 Cost Projections (at 1000 requests/day):")
            print("-" * 60)
            print(f"Daily: ${daily_cost:.2f}")
            print(f"Monthly: ${monthly_cost:.2f}")
            print("-" * 60)
        
        print("\n📥 Response:")
        print("-" * 60)
        if hasattr(response, 'generations'):
            for generation in response.generations:
                for gen in generation:
                    response_text = gen.text
                    print(response_text[:300] + "..." if len(response_text) > 300 else response_text)
        print("-" * 60)


def main():
    print("=" * 60)
    print("STAGE 4: Cost Tracking")
    print("=" * 60)
    
    # Initialize the LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env file")
        return
    
    model_name = "gpt-3.5-turbo"
    
    # Create the cost tracking callback
    callback = CostTrackingCallback(model_name=model_name)
    
    print(f"\n✓ CostTrackingCallback initialized for {model_name}")
    print(f"✓ Pricing: ${CostTrackingCallback.PRICING[model_name]['prompt']}/1K prompt tokens")
    print(f"           ${CostTrackingCallback.PRICING[model_name]['completion']}/1K completion tokens")
    
    # Initialize LLM with callback
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7,
        openai_api_key=api_key,
        callbacks=[callback]
    )
    
    print("✓ ChatOpenAI model initialized with cost tracking")
    
    # Create prompt template
    template = """You are a helpful AI assistant.

User question: {user_input}

Please provide a clear answer."""
    
    prompt = PromptTemplate(
        input_variables=["user_input"],
        template=template
    )
    
    # Build chain
    chain = prompt | llm | StrOutputParser()
    
    print("✓ Chain built with cost tracking enabled")
    
    # Test with multiple questions to show cumulative cost
    questions = [
        "What is LLM observability?",
        "Why is cost tracking important for production LLM applications?",
        "How can we optimize token usage to reduce costs?"
    ]
    
    for question in questions:
        print(f"\n{'=' * 60}")
        print(f"📝 Question: {question}")
        print(f"{'=' * 60}")
        
        response = chain.invoke(
            {"user_input": question},
            config={"callbacks": [callback]}
        )
    
    print("\n" + "=" * 60)
    print("✅ Stage 4 Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- Token usage directly translates to API costs")
    print("- Tracking costs helps with budget planning")
    print("- Different models have different pricing structures")
    print("- Cumulative tracking shows total spend over time")
    print("- Cost projections help estimate production expenses")
    print("\nCost Optimization Tips:")
    print("- Use gpt-3.5-turbo for simple tasks (cheaper)")
    print("- Optimize prompts to reduce token count")
    print("- Cache responses when possible")
    print("- Set token limits (max_tokens parameter)")
    print("\nNext: Run step5_error_handling.py to add reliability features")

if __name__ == "__main__":
    main()
