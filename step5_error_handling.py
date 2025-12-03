"""
STAGE 5: Error Handling and Retry Logic
=======================================
This script adds production-ready error handling and retry logic.

What you'll learn:
- Common OpenAI API errors (rate limits, timeouts, API errors)
- How to implement retry logic with exponential backoff
- How to track error rates in observability
- Production reliability patterns

Essential for building robust LLM applications.
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

# Import OpenAI exceptions
from openai import (
    APIError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError
)

# Import tenacity for retry logic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

# Load environment variables
load_dotenv()


class ErrorTrackingCallback(BaseCallbackHandler):
    """
    Callback handler with error tracking capabilities.
    
    Tracks:
    - Successful requests
    - Failed requests
    - Error types
    - Retry attempts
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        super().__init__()
        self.model_name = model_name
        self.start_time = None
        self.end_time = None
        self.success_count = 0
        self.error_count = 0
        self.error_types = {}
        self.retry_count = 0
    
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> None:
        """Called when LLM starts processing."""
        self.start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n" + "=" * 60)
        print(f"🕐 Start Time: {timestamp}")
        print("=" * 60)
    
    def on_llm_end(
        self, 
        response: Any, 
        **kwargs: Any
    ) -> None:
        """Called when LLM finishes successfully."""
        self.end_time = time.time()
        latency = self.end_time - self.start_time
        self.success_count += 1
        
        print("\n✅ Request Successful")
        print(f"⏱️  Latency: {latency:.3f} seconds")
        
        self.print_stats()
    
    def on_llm_error(
        self, 
        error: Exception, 
        **kwargs: Any
    ) -> None:
        """Called when LLM encounters an error."""
        self.error_count += 1
        error_type = type(error).__name__
        
        # Track error types
        if error_type not in self.error_types:
            self.error_types[error_type] = 0
        self.error_types[error_type] += 1
        
        print(f"\n❌ LLM Error: {error_type}")
        print(f"   Message: {str(error)}")
        
        self.print_stats()
    
    def print_stats(self):
        """Print current error tracking statistics."""
        total_requests = self.success_count + self.error_count
        success_rate = (self.success_count / total_requests * 100) if total_requests > 0 else 0
        
        print("\n📊 Reliability Metrics:")
        print("-" * 60)
        print(f"✅ Successful Requests: {self.success_count}")
        print(f"❌ Failed Requests: {self.error_count}")
        print(f"📈 Success Rate: {success_rate:.2f}%")
        
        if self.error_types:
            print("\n🔍 Error Breakdown:")
            for error_type, count in self.error_types.items():
                print(f"   {error_type}: {count}")
        
        print("-" * 60)


class ResilientLLMWrapper:
    """
    Wrapper class that adds retry logic to LLM calls.
    
    Features:
    - Automatic retry with exponential backoff
    - Specific handling for different error types
    - Configurable retry attempts
    """
    
    def __init__(self, chain, callback, max_retries: int = 3):
        self.chain = chain
        self.callback = callback
        self.max_retries = max_retries
    
    @retry(
        # Retry on specific OpenAI exceptions
        retry=retry_if_exception_type((
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            APIError
        )),
        # Stop after N attempts
        stop=stop_after_attempt(3),
        # Exponential backoff: wait 2^x * 1 seconds between retries
        wait=wait_exponential(multiplier=1, min=2, max=10),
        # Don't reraise the exception immediately
        reraise=True
    )
    def invoke_with_retry(self, input_data: Dict[str, Any]) -> str:
        """
        Invoke the chain with automatic retry logic.
        
        Args:
            input_data: Input dictionary for the chain
        
        Returns:
            Response string from the LLM
        
        Raises:
            Exception: If all retry attempts fail
        """
        print("🔄 Attempting LLM call...")
        
        try:
            response = self.chain.invoke(
                input_data,
                config={"callbacks": [self.callback]}
            )
            return response
        
        except RateLimitError as e:
            print("⚠️  Rate limit hit - will retry with backoff...")
            self.callback.retry_count += 1
            raise
        
        except APITimeoutError as e:
            print("⚠️  Request timeout - will retry...")
            self.callback.retry_count += 1
            raise
        
        except APIConnectionError as e:
            print("⚠️  Connection error - will retry...")
            self.callback.retry_count += 1
            raise
        
        except APIError as e:
            print("⚠️  API error - will retry...")
            self.callback.retry_count += 1
            raise
        
        except Exception as e:
            print(f"❌ Unexpected error: {type(e).__name__} - {str(e)}")
            raise


def main():
    print("=" * 60)
    print("STAGE 5: Error Handling and Retry Logic")
    print("=" * 60)
    
    # Initialize the LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env file")
        return
    
    model_name = "gpt-3.5-turbo"
    
    # Create error tracking callback
    callback = ErrorTrackingCallback(model_name=model_name)
    
    print(f"\n✓ ErrorTrackingCallback initialized")
    
    # Initialize LLM with callback
    # Add timeout and max_retries settings
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7,
        openai_api_key=api_key,
        callbacks=[callback],
        request_timeout=30,  # 30 second timeout
        max_retries=0  # We handle retries ourselves
    )
    
    print("✓ ChatOpenAI model initialized with error tracking")
    
    # Create prompt template
    template = """You are a helpful AI assistant.

User question: {user_input}

Please provide a brief answer."""
    
    prompt = PromptTemplate(
        input_variables=["user_input"],
        template=template
    )
    
    # Build chain
    chain = prompt | llm | StrOutputParser()
    
    # Wrap chain with retry logic
    resilient_chain = ResilientLLMWrapper(chain, callback, max_retries=3)
    
    print("✓ Chain wrapped with retry logic")
    print("✓ Retry configuration:")
    print("   - Max attempts: 3")
    print("   - Exponential backoff: 2^x seconds")
    print("   - Timeout: 30 seconds")
    
    # Test with questions
    questions = [
        "What are common API errors in LLM applications?",
        "How does exponential backoff help with rate limiting?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 60}")
        print(f"Question {i}: {question}")
        print(f"{'=' * 60}")
        
        try:
            response = resilient_chain.invoke_with_retry(
                {"user_input": question}
            )
            
            print("\n📥 Response:")
            print("-" * 60)
            print(response)
            print("-" * 60)
        
        except Exception as e:
            print(f"\n❌ Request failed after all retry attempts")
            print(f"   Final error: {type(e).__name__}")
            print(f"   Message: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ Stage 5 Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- Always implement retry logic for production LLM apps")
    print("- Use exponential backoff to respect rate limits")
    print("- Track error rates for SLO monitoring")
    print("- Different errors need different handling strategies")
    print("- Set appropriate timeouts to prevent hanging requests")
    print("\nCommon OpenAI API Errors:")
    print("- RateLimitError: Too many requests (retry with backoff)")
    print("- APITimeoutError: Request took too long (retry)")
    print("- APIConnectionError: Network issue (retry)")
    print("- APIError: Server-side error (retry)")
    print("\nNext: Run step6_prometheus_metrics.py to add metrics export")

if __name__ == "__main__":
    main()
