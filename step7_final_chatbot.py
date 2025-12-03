"""
STAGE 7: Final Integration - Complete Chatbot with Full Observability
======================================================================
This script brings everything together into a production-ready chatbot.

Features:
- Interactive chatbot loop
- Prometheus metrics export
- Cost tracking
- Error handling with retries
- Full observability pipeline
- Session management

This demonstrates a complete, production-ready LLM application with
comprehensive observability.
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

# Prometheus metrics
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    start_http_server
)

# Retry logic
from openai import (
    APIError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

# Load environment variables
load_dotenv()


# ============================================================================
# Prometheus Metrics Definition
# ============================================================================

llm_request_count = Counter(
    'llm_request_count',
    'Total number of LLM requests',
    ['model', 'status']
)

llm_error_count = Counter(
    'llm_error_count',
    'Total number of LLM errors',
    ['model', 'error_type']
)

llm_request_duration = Histogram(
    'llm_request_duration_seconds',
    'LLM request duration in seconds',
    ['model'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total number of tokens used',
    ['model', 'token_type']
)

llm_cost_usd = Gauge(
    'llm_cost_usd_total',
    'Cumulative estimated cost in USD',
    ['model']
)

llm_session_duration = Gauge(
    'llm_session_duration_seconds',
    'Current session duration',
    ['session_id']
)


# ============================================================================
# Complete Observability Callback
# ============================================================================

class ComprehensiveCallback(BaseCallbackHandler):
    """
    Production-ready callback with full observability features.
    
    Features:
    - Prometheus metrics export
    - Cost tracking
    - Error tracking
    - Latency monitoring
    - Session management
    """
    
    PRICING = {
        "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03}
    }
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", session_id: str = "default"):
        super().__init__()
        self.model_name = model_name
        self.session_id = session_id
        self.start_time = None
        self.cumulative_cost = 0.0
        self.session_start = time.time()
        self.request_number = 0
    
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> None:
        """Track request start."""
        self.start_time = time.time()
        self.request_number += 1
        
        # Update session duration
        session_duration = time.time() - self.session_start
        llm_session_duration.labels(session_id=self.session_id).set(session_duration)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{timestamp}] 🤖 Processing your request...")
    
    def on_llm_end(
        self, 
        response: Any, 
        **kwargs: Any
    ) -> None:
        """Update all metrics on successful completion."""
        latency = time.time() - self.start_time
        
        # Extract token usage
        prompt_tokens = 0
        completion_tokens = 0
        
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            prompt_tokens = token_usage.get('prompt_tokens', 0)
            completion_tokens = token_usage.get('completion_tokens', 0)
        
        # Calculate cost
        pricing = self.PRICING.get(self.model_name, self.PRICING["gpt-3.5-turbo"])
        request_cost = (
            (prompt_tokens / 1000.0) * pricing["prompt"] +
            (completion_tokens / 1000.0) * pricing["completion"]
        )
        self.cumulative_cost += request_cost
        
        # Update Prometheus metrics
        llm_request_count.labels(model=self.model_name, status='success').inc()
        llm_request_duration.labels(model=self.model_name).observe(latency)
        llm_tokens_total.labels(model=self.model_name, token_type='prompt').inc(prompt_tokens)
        llm_tokens_total.labels(model=self.model_name, token_type='completion').inc(completion_tokens)
        llm_cost_usd.labels(model=self.model_name).set(self.cumulative_cost)
        
        # Log metrics (condensed format)
        print(f"   ⏱️  {latency:.2f}s | 🎯 {prompt_tokens + completion_tokens} tokens | 💰 ${request_cost:.6f}")
    
    def on_llm_error(
        self, 
        error: Exception, 
        **kwargs: Any
    ) -> None:
        """Track errors."""
        error_type = type(error).__name__
        
        llm_request_count.labels(model=self.model_name, status='error').inc()
        llm_error_count.labels(model=self.model_name, error_type=error_type).inc()
        
        print(f"   ❌ Error: {error_type}")
    
    def print_session_stats(self):
        """Print session statistics."""
        session_duration = time.time() - self.session_start
        avg_cost = self.cumulative_cost / self.request_number if self.request_number > 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 Session Statistics")
        print("=" * 60)
        print(f"🕐 Session Duration: {session_duration:.1f} seconds")
        print(f"🔢 Total Requests: {self.request_number}")
        print(f"💰 Total Cost: ${self.cumulative_cost:.6f}")
        print(f"📈 Average Cost/Request: ${avg_cost:.6f}")
        print("=" * 60)


# ============================================================================
# Resilient Chain Wrapper with Retry Logic
# ============================================================================

class ResilientChatbot:
    """Chatbot with automatic retry logic."""
    
    def __init__(self, chain, callback):
        self.chain = chain
        self.callback = callback
    
    @retry(
        retry=retry_if_exception_type((
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            APIError
        )),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def chat(self, user_input: str) -> str:
        """Send message with automatic retry."""
        try:
            response = self.chain.invoke(
                {"user_input": user_input},
                config={"callbacks": [self.callback]}
            )
            return response
        except Exception as e:
            print(f"   ⚠️  Retrying after error: {type(e).__name__}")
            raise


# ============================================================================
# Main Application
# ============================================================================

def main():
    print("=" * 60)
    print("🤖 LLM CHATBOT WITH FULL OBSERVABILITY")
    print("=" * 60)
    
    # Start Prometheus metrics server
    metrics_port = 8000
    print(f"\n🚀 Starting metrics server on port {metrics_port}")
    print(f"📊 Metrics: http://localhost:{metrics_port}/metrics")
    start_http_server(metrics_port)
    
    # Load API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY not found in .env file")
        print("Please create a .env file with your OpenAI API key")
        return
    
    print("✓ API key loaded")
    
    # Initialize components
    model_name = "gpt-3.5-turbo"
    session_id = f"session_{int(time.time())}"
    
    callback = ComprehensiveCallback(model_name=model_name, session_id=session_id)
    print(f"✓ Observability initialized (Session: {session_id})")
    
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7,
        openai_api_key=api_key,
        callbacks=[callback],
        request_timeout=30,
        max_retries=0  # We handle retries
    )
    print(f"✓ LLM initialized ({model_name})")
    
    # Create chatbot prompt
    template = """You are a helpful AI assistant with expertise in software engineering and observability.

Previous context: This is a conversational chatbot demonstrating LLM observability patterns.

User: {user_input}"""
    
    prompt = PromptTemplate(
        input_variables=["user_input"],
        template=template
    )
    
    # Build chain
    chain = prompt | llm | StrOutputParser()
    
    # Wrap with retry logic
    chatbot = ResilientChatbot(chain, callback)
    
    print("✓ Chat chain built with full observability")
    print("\n" + "=" * 60)
    print("Ready to chat! Type 'quit', 'exit', or 'bye' to end session")
    print("Type 'stats' to see session statistics")
    print("=" * 60)
    
    # Example conversation (you can replace with input loop)
    print("\n📝 Demo Conversation:")
    print("=" * 60)
    
    demo_questions = [
        "What is observability in the context of LLMs?",
        "How can I track costs in my LLM application?",
        "What metrics should I monitor?"
    ]
    
    for question in demo_questions:
        print(f"\n👤 User: {question}")
        
        try:
            response = chatbot.chat(question)
            print(f"🤖 Assistant: {response}\n")
            print("-" * 60)
        
        except Exception as e:
            print(f"❌ Fatal error after all retries: {str(e)}")
            break
    
    # Interactive loop (commented out for demo - uncomment for real use)
    """
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'stats':
                callback.print_session_stats()
                continue
            
            response = chatbot.chat(user_input)
            print(f"\n🤖 Assistant: {response}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
    """
    
    # Print final statistics
    callback.print_session_stats()
    
    print("\n" + "=" * 60)
    print("✅ Stage 7 Complete!")
    print("=" * 60)
    print("\n🎉 Congratulations! You've completed all 7 stages!")
    print("\nKey Features Demonstrated:")
    print("✓ Interactive chatbot")
    print("✓ Prometheus metrics export")
    print("✓ Cost tracking")
    print("✓ Error handling with retries")
    print("✓ Latency monitoring")
    print("✓ Session management")
    
    print("\n📊 Production Checklist:")
    print("□ Set up Prometheus scraping")
    print("□ Create Grafana dashboards")
    print("□ Configure alerting rules")
    print("□ Set up cost budgets")
    print("□ Implement rate limiting")
    print("□ Add authentication")
    print("□ Deploy to production environment")
    
    print("\n🔍 View Live Metrics:")
    print(f"   http://localhost:{metrics_port}/metrics")
    
    print("\n⚠️  Server is still running. Press Ctrl+C to stop...")
    
    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")


if __name__ == "__main__":
    main()
