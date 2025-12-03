# Quick Start Guide

## Project Overview

This project demonstrates LLM observability use cases in a step-by-step manner using LangChain and OpenAI.

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Basic understanding of Python and LLMs

## Setup Instructions

### 1. Navigate to the project directory

```bash
cd LLM-Observability-examples
```

### 2. Create and activate virtual environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Or activate (Windows)
# venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API key

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-actual-api-key-here
```

## Running the Stages

Execute the files in order to learn each concept:

### Stage 1: Basic Chatbot
```bash
python step1_basic_chatbot.py
```
**Learn:** LangChain + OpenAI basics, simple prompt → response flow

### Stage 2: Prompt Templates
```bash
python step2_prompt_template.py
```
**Learn:** PromptTemplate usage, chain building, variable substitution

### Stage 3: Basic Callback
```bash
python step3_basic_callback.py
```
**Learn:** Custom callbacks, capturing prompts/responses, token tracking, latency measurement

### Stage 4: Cost Tracking
```bash
python step4_cost_tracking.py
```
**Learn:** Token usage analysis, cost calculation, budget projection

### Stage 5: Error Handling
```bash
python step5_error_handling.py
```
**Learn:** Retry logic, exponential backoff, error tracking, reliability patterns

### Stage 6: Prometheus Metrics
```bash
python step6_prometheus_metrics.py
```
**Learn:** Metrics export, Prometheus integration, monitoring setup

Access metrics at: `http://localhost:8000/metrics`

### Stage 7: Final Integration
```bash
python step7_final_chatbot.py
```
**Learn:** Complete chatbot with all observability features integrated

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Application                          │
├─────────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  Prompt    │→ │    LLM     │→ │   Output   │            │
│  │  Template  │  │   Chain    │  │   Parser   │            │
│  └────────────┘  └────────────┘  └────────────┘            │
│         ↓              ↓               ↓                     │
│  ┌──────────────────────────────────────────────┐          │
│  │         Callback Handler (Observability)      │          │
│  │  • Token Tracking   • Cost Calculation        │          │
│  │  • Latency Metrics  • Error Tracking          │          │
│  └──────────────────────────────────────────────┘          │
│         ↓                                                    │
│  ┌──────────────────────────────────────────────┐          │
│  │        Prometheus Metrics Exporter            │          │
│  │  Counter, Histogram, Gauge metrics            │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
         ↓                           ↓
┌─────────────────┐        ┌─────────────────┐
│   Prometheus    │        │     Grafana     │
│  (Metrics DB)   │   →    │  (Visualization) │
└─────────────────┘        └─────────────────┘
```

## Key Observability Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `llm_request_count` | Counter | Total requests (by model and status) |
| `llm_request_duration_seconds` | Histogram | Latency distribution |
| `llm_tokens_total` | Counter | Token usage (prompt vs completion) |
| `llm_cost_usd_total` | Gauge | Cumulative estimated cost |
| `llm_error_count` | Counter | Errors by type |
| `llm_session_duration_seconds` | Gauge | Session duration |

## Prometheus Configuration

Add this to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'llm-chatbot'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
```

## Grafana Dashboard Queries

### Request Rate
```promql
rate(llm_request_count{status="success"}[5m])
```

### P95 Latency
```promql
histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))
```

### Token Usage Rate
```promql
rate(llm_tokens_total[5m])
```

### Error Rate
```promql
rate(llm_request_count{status="error"}[5m])
```

### Total Cost
```promql
llm_cost_usd_total
```

## Production Deployment Checklist

- [ ] Set up Prometheus scraping
- [ ] Configure Grafana dashboards
- [ ] Set up alerting rules (high latency, error rate, cost thresholds)
- [ ] Implement rate limiting
- [ ] Add authentication/authorization
- [ ] Configure retry policies
- [ ] Set up logging aggregation
- [ ] Implement circuit breakers
- [ ] Add health check endpoints
- [ ] Set up cost budgets and alerts

## Troubleshooting

### "OPENAI_API_KEY not found"
- Ensure `.env` file exists in the project directory
- Check that the API key is correctly formatted
- Verify the `.env` file is in the same directory as the script

### "Import could not be resolved"
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version (3.8+ required)

### "Port 8000 already in use"
- Change the port in step6 or step7 scripts
- Or stop the process using port 8000

### High costs
- Use `gpt-3.5-turbo` instead of `gpt-4`
- Reduce `max_tokens` parameter
- Implement caching for common queries
- Set daily/monthly cost limits

## Next Steps

1. **Integrate with your application:** Use these patterns in your production LLM app
2. **Set up monitoring:** Deploy Prometheus and Grafana
3. **Add custom metrics:** Track domain-specific metrics
4. **Implement alerting:** Set up alerts for SLO violations
5. **Optimize costs:** Use insights to reduce token usage
6. **Scale:** Deploy to cloud infrastructure

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

## License

MIT
