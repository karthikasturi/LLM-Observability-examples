# LLM Observability Demo with LangChain

A step-by-step tutorial demonstrating LLM observability patterns using LangChain and OpenAI.

## Prerequisites

- Python 3.8+
- OpenAI API key

## Setup

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate  # On Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

## Stages Overview

### Stage 1: Basic Chatbot
**File:** `step1_basic_chatbot.py`

Learn the fundamentals of LangChain with OpenAI integration.
```bash
python step1_basic_chatbot.py
```

### Stage 2: Prompt Templates
**File:** `step2_prompt_template.py`

Understand prompt engineering with LangChain's PromptTemplate.
```bash
python step2_prompt_template.py
```

### Stage 3: Basic Callback
**File:** `step3_basic_callback.py`

Capture prompts, responses, tokens, and latency using callbacks.
```bash
python step3_basic_callback.py
```

### Stage 4: Cost Tracking
**File:** `step4_cost_tracking.py`

Track token usage and estimate OpenAI API costs.
```bash
python step4_cost_tracking.py
```

### Stage 5: Error Handling
**File:** `step5_error_handling.py`

Implement retry logic and error handling for production reliability.
```bash
python step5_error_handling.py
```

### Stage 6: Prometheus Metrics
**File:** `step6_prometheus_metrics.py`

Export observability metrics to Prometheus for monitoring.
```bash
python step6_prometheus_metrics.py
```

Access metrics at: http://localhost:8000/metrics

### Stage 7: Final Integration
**File:** `step7_final_chatbot.py`

Complete chatbot with full observability pipeline.
```bash
python step7_final_chatbot.py
```

## Key Concepts Covered

- ✅ LangChain basics
- ✅ Prompt templates and chains
- ✅ Custom callback handlers
- ✅ Token usage tracking
- ✅ Cost estimation
- ✅ Error handling and retries
- ✅ Prometheus metrics integration
- ✅ Production-ready observability

## Monitoring with Prometheus & Grafana

After running Stage 6 or 7, you can:

1. **Scrape metrics with Prometheus:**
   Add this to your `prometheus.yml`:
   ```yaml
   scrape_configs:
     - job_name: 'llm-chatbot'
       static_configs:
         - targets: ['localhost:8000']
   ```

2. **Visualize in Grafana:**
   - Import Prometheus datasource
   - Create dashboards for:
     - Request count over time
     - Token usage trends
     - Latency percentiles
     - Error rates

## License

MIT
