"""
STAGE 3: Basic Callback for Observability
=========================================
This script introduces callback handlers to capture LLM interactions.

What you'll learn:
- How to create a custom callback handler
- How to capture prompts, responses, and timestamps
- How to track token usage
- How to calculate latency

This is the foundation for LLM observability.
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


class ObservabilityCallback(BaseCallbackHandler):
    """
    Custom callback handler to capture LLM observability metrics.
    
    This handler hooks into the LangChain lifecycle to capture:
    - Prompts sent to the LLM
    - Responses received
    - Token usage
    - Latency
    """
    
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.end_time = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
    
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> None:
        """
        Called when LLM starts processing.
        Captures the prompt and start timestamp.
        """
        self.start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n" + "=" * 60)
        print(f"🕐 LLM Start Time: {timestamp}")
        print("=" * 60)
        print("\n📤 Prompt sent to LLM:")
        print("-" * 60)
        for i, prompt in enumerate(prompts, 1):
            print(f"Prompt {i}:")
            print(prompt)
        print("-" * 60)
    
    def on_llm_end(
        self, 
        response: Any, 
        **kwargs: Any
    ) -> None:
        """
        Called when LLM finishes processing.
        Captures the response, token usage, and calculates latency.
        """
        self.end_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate latency
        latency = self.end_time - self.start_time
        
        # Extract token usage from response metadata
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            self.prompt_tokens = token_usage.get('prompt_tokens', 0)
            self.completion_tokens = token_usage.get('completion_tokens', 0)
            self.total_tokens = token_usage.get('total_tokens', 0)
        
        print("\n" + "=" * 60)
        print(f"🕐 LLM End Time: {timestamp}")
        print("=" * 60)
        
        print("\n📊 Observability Metrics:")
        print("-" * 60)
        print(f"⏱️  Latency: {latency:.3f} seconds")
        print(f"🎯 Prompt Tokens: {self.prompt_tokens}")
        print(f"💬 Completion Tokens: {self.completion_tokens}")
        print(f"📈 Total Tokens: {self.total_tokens}")
        print("-" * 60)
        
        print("\n📥 Response from LLM:")
        print("-" * 60)
        if hasattr(response, 'generations'):
            for generation in response.generations:
                for gen in generation:
                    print(gen.text)
        print("-" * 60)
    
    def on_llm_error(
        self, 
        error: Exception, 
        **kwargs: Any
    ) -> None:
        """
        Called when LLM encounters an error.
        Useful for tracking failures.
        """
        print(f"\n❌ LLM Error: {error}")


def main():
    print("=" * 60)
    print("STAGE 3: Basic Callback for Observability")
    print("=" * 60)
    
    # Initialize the LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env file")
        return
    
    # Create the callback handler
    callback = ObservabilityCallback()
    
    print("\n✓ ObservabilityCallback initialized")
    
    # Initialize LLM with callback
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=api_key,
        callbacks=[callback]  # Attach the callback handler
    )
    
    print("✓ ChatOpenAI model initialized with callback")
    
    # Create prompt template
    template = """You are a helpful AI assistant.

User question: {user_input}

Please provide a brief answer (2-3 sentences)."""
    
    prompt = PromptTemplate(
        input_variables=["user_input"],
        template=template
    )
    
    # Build chain
    chain = prompt | llm | StrOutputParser()
    
    print("✓ Chain built with observability enabled")
    
    # Test with a question
    question = "What is the importance of monitoring LLM applications in production?"
    
    print(f"\n📝 User Question: {question}")
    
    # Invoke the chain - callbacks will automatically fire
    response = chain.invoke(
        {"user_input": question},
        config={"callbacks": [callback]}
    )
    
    print("\n✅ Stage 3 Complete!")
    print("\nKey Takeaways:")
    print("- Callbacks hook into the LLM lifecycle")
    print("- on_llm_start captures prompts and start time")
    print("- on_llm_end captures responses, tokens, and calculates latency")
    print("- This is the foundation for all observability patterns")
    print("\nNext: Run step4_cost_tracking.py to add cost estimation")

if __name__ == "__main__":
    main()
