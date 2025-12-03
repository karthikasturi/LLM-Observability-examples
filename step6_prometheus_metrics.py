"""
STAGE 6: Prometheus Metrics Exporter
====================================
This script adds Prometheus metrics export for production monitoring.

What you'll learn:
- How to expose a /metrics HTTP endpoint
- What metrics to track for LLM observability
- How to use Prometheus client library
- Integration with monitoring stacks (Prometheus + Grafana)

Metrics exposed:
- llm_request_count: Total number of requests
- llm_request_duration_seconds: Histogram of request latencies
- llm_tokens_used_total: Total tokens consumed
- llm_error_count: Total number of errors
- llm_cost_usd_total: Estimated cumulative cost
"""

import os
import time
from datetime import datetime
from typing import Any, Dict, List
from threading import Thread
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler

# Prometheus client imports
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    start_http_server,
    REGISTRY
)

# Load environment variables
load_dotenv()


# Define Prometheus metrics
# Counter: A cumulative metric that only increases
llm_request_count = Counter(
    'llm_request_count',
    'Total number of LLM requests',
    ['model', 'status']  # Labels for filtering
)

llm_error_count = Counter(
    'llm_error_count',
    'Total number of LLM errors',
    ['model', 'error_type']
)

# Histogram: Distribution of observations (for latency percentiles)
llm_request_duration = Histogram(
    'llm_request_duration_seconds',
    'LLM request duration in seconds',
    ['model'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)  # Latency buckets
)

# Counter for tokens
llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total number of tokens used',
    ['model', 'token_type']  # token_type: prompt or completion
)

# Gauge: Current value that can go up or down
llm_cost_usd = Gauge(
    'llm_cost_usd_total',
    'Cumulative estimated cost in USD',
    ['model']
)


class PrometheusCallback(BaseCallbackHandler):
    """
    Callback handler that exports metrics to Prometheus.
    
    Tracks:
    - Request counts (success/failure)
    - Latency distributions
    - Token usage
    - Error types
    - Estimated costs
    """
    
    # OpenAI Pricing (per 1K tokens)
    PRICING = {
        "gpt-3.5-turbo": {
            "prompt": 0.0015,
            "completion": 0.002
        },
        "gpt-4": {
            "prompt": 0.03,
            "completion": 0.06
        }
    }
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        super().__init__()
        self.model_name = model_name
        self.start_time = None
        self.cumulative_cost = 0.0
    
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> None:
        """Record start time for latency calculation."""
        self.start_time = time.time()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n🕐 [{timestamp}] Starting LLM request...")
    
    def on_llm_end(
        self, 
        response: Any, 
        **kwargs: Any
    ) -> None:
        """Update all Prometheus metrics on successful completion."""
        # Calculate latency
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
        
        # Log to console
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"✅ [{timestamp}] Request completed successfully")
        print(f"   ⏱️  Latency: {latency:.3f}s")
        print(f"   🎯 Tokens: {prompt_tokens} prompt + {completion_tokens} completion")
        print(f"   💰 Cost: ${request_cost:.6f} (Total: ${self.cumulative_cost:.6f})")
    
    def on_llm_error(
        self, 
        error: Exception, 
        **kwargs: Any
    ) -> None:
        """Track errors in Prometheus."""
        error_type = type(error).__name__
        
        # Update metrics
        llm_request_count.labels(model=self.model_name, status='error').inc()
        llm_error_count.labels(model=self.model_name, error_type=error_type).inc()
        
        # Log to console
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"❌ [{timestamp}] Request failed: {error_type}")


def start_metrics_server(port: int = 8000):
    """
    Start Prometheus metrics HTTP server.
    
    Args:
        port: Port to expose metrics on (default: 8000)
    """
    print(f"\n🚀 Starting Prometheus metrics server on port {port}")
    print(f"📊 Metrics available at: http://localhost:{port}/metrics")
    print("-" * 60)
    
    # Start the HTTP server in a background thread
    start_http_server(port)


def main():
    print("=" * 60)
    print("STAGE 6: Prometheus Metrics Export")
    print("=" * 60)
    
    # Start Prometheus metrics server
    start_metrics_server(port=8000)
    
    # Initialize the LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env file")
        return
    
    model_name = "gpt-3.5-turbo"
    
    # Create Prometheus callback
    callback = PrometheusCallback(model_name=model_name)
    
    print(f"\n✓ PrometheusCallback initialized for {model_name}")
    
    # Initialize LLM with callback
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7,
        openai_api_key=api_key,
        callbacks=[callback]
    )
    
    print("✓ ChatOpenAI model initialized with Prometheus metrics")
    
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
    
    print("✓ Chain built with metrics export enabled")
    
    # Test with multiple questions
    questions = [
        "What is Prometheus?",
        "Why are histograms useful for latency tracking?",
        "What's the difference between Counter and Gauge metrics?",
    ]
    
    print("\n" + "=" * 60)
    print("Sending test requests...")
    print("=" * 60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 Question {i}: {question}")
        
        try:
            response = chain.invoke(
                {"user_input": question},
                config={"callbacks": [callback]}
            )
            
            print(f"💬 Response preview: {response[:100]}...")
        
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ Stage 6 Complete!")
    print("=" * 60)
    print("\n📊 Prometheus Metrics Available:")
    print("-" * 60)
    print("1. llm_request_count - Total requests by model and status")
    print("2. llm_request_duration_seconds - Latency histogram")
    print("3. llm_tokens_total - Token usage by type")
    print("4. llm_error_count - Errors by type")
    print("5. llm_cost_usd_total - Cumulative cost estimate")
    print("-" * 60)
    
    print("\n🔍 View Metrics:")
    print(f"   curl http://localhost:8000/metrics")
    print("\n🎨 Grafana Dashboard Setup:")
    print("   1. Add Prometheus datasource (http://localhost:9090)")
    print("   2. Create dashboard with queries:")
    print("      - rate(llm_request_count[5m])")
    print("      - histogram_quantile(0.95, llm_request_duration_seconds)")
    print("      - rate(llm_tokens_total[5m])")
    print("      - llm_cost_usd_total")
    
    print("\n⚠️  Server will keep running to expose metrics...")
    print("   Press Ctrl+C to stop")
    print("\nNext: Run step7_final_chatbot.py for complete integration")
    
    # Keep the script running to serve metrics
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down metrics server...")

if __name__ == "__main__":
    main()
