# LLM Observability with LangSmith

A step-by-step tutorial demonstrating **LangSmith observability** for LLM applications using LangChain and OpenAI.

This project focuses on **LangSmith's capabilities** for:
- 🔍 Automatic tracing of LLM interactions
- 📊 Custom metadata and tagging
- 🧪 LLM-as-a-judge evaluations
- 📦 Dataset-based testing
- 📈 Run comparisons and analytics

**No Prometheus or infrastructure monitoring** - pure LangSmith observability!

## Prerequisites

- Python 3.8+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- LangSmith API key ([Sign up here](https://smith.langchain.com/))

## Quick Start

### 1. Clone and Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file with your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add:
```
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=llm-observability-demo
```

### 3. Run the Stages

```bash
# Stage 1: Basic chatbot (no observability)
python step1_basic_chatbot.py

# Stage 2: Add prompt templates
python step2_prompt_template.py

# Stage 3: Enable LangSmith tracing
python step3_langsmith_tracing.py

# Stage 4: Add metadata and tags
python step4_tracing_metadata.py

# Stage 5: LLM-as-a-judge evaluation
python step5_evaluations.py

# Stage 6: Dataset-based testing
python step6_dataset_testing.py

# Stage 7: Complete integration
python step7_full_chatbot.py
```

## 📚 Stages Overview

### Stage 1: Basic Chatbot
**File:** `step1_basic_chatbot.py`

Simple interactive chatbot using LangChain + OpenAI. No observability yet.

**Learn:**
- LangChain basics
- ChatOpenAI model usage
- Interactive chat loop

---

### Stage 2: Prompt Templates and Chains
**File:** `step2_prompt_template.py`

Build reusable prompts and chains using LangChain's PromptTemplate.

**Learn:**
- Prompt engineering with templates
- Variable substitution with `{placeholders}`
- Chain construction with pipe (`|`) operator

---

### Stage 3: Enable LangSmith Tracing
**File:** `step3_langsmith_tracing.py`

Enable automatic tracing with LangSmith. View all traces in LangSmith UI.

**Learn:**
- Setting up LangSmith
- `@traceable` decorator for custom functions
- Viewing traces in LangSmith dashboard

---

### Stage 4: Rich Tracing with Metadata and Tags
**File:** `step4_tracing_metadata.py`

Add custom metadata and tags to organize and filter traces.

**Learn:**
- Adding custom metadata (environment, version, etc.)
- Tagging runs for categorization
- Nested traced functions
- Filtering and searching in LangSmith

---

### Stage 5: LLM-as-a-Judge Evaluation
**File:** `step5_evaluations.py`

Use an LLM to evaluate other LLM responses automatically.

**Learn:**
- LLM-based evaluation
- Scoring on relevance, hallucination risk, tone
- Evaluation traces in LangSmith
- Quality assessment automation

---

### Stage 6: Dataset-Based Testing
**File:** `step6_dataset_testing.py`

Create test datasets and run batch evaluations for regression testing.

**Learn:**
- Creating LangSmith datasets
- Batch evaluation with `run_on_dataset()`
- Tracking test results over time
- Dataset-driven development

---

### Stage 7: Full Integration
**File:** `step7_full_chatbot.py`

Complete chatbot combining all LangSmith features into one application.

**Learn:**
- Integrating tracing + metadata + evaluation
- Production-ready structure
- End-to-end observability pipeline

---

## 🔍 LangSmith Features Demonstrated

| Feature | Stages |
|---------|--------|
| Automatic tracing | 3-7 |
| Custom `@traceable` functions | 3-7 |
| Metadata & tags | 4-7 |
| Nested traces | 4-7 |
| LLM-as-a-judge evaluation | 5, 7 |
| Dataset management | 6 |
| Batch testing | 6 |
| Run comparisons | All |

## 📊 Viewing Your Traces

1. Go to [https://smith.langchain.com/](https://smith.langchain.com/)
2. Select your project (e.g., `llm-observability-demo`)
3. Browse runs and traces
4. Filter by metadata or tags
5. Compare performance across runs
6. Drill into nested trace hierarchies

## 🎯 Key Concepts

### Tracing
Every LLM interaction is automatically logged with:
- Input prompts
- Output responses
- Token usage
- Latency
- Model parameters

### Metadata
Add structured context to traces:
```python
@traceable(metadata={"environment": "prod", "version": "1.0"})
```

### Tags
Categorize runs for easy filtering:
```python
@traceable(tags=["customer-support", "urgent"])
```

### Evaluation
Automatically assess response quality:
- Relevance scoring
- Hallucination detection
- Tone analysis
- Custom evaluation criteria

### Datasets
Systematic testing with reusable test cases:
- Regression testing
- A/B testing
- Performance tracking
- Quality benchmarking

## 🚀 Production Best Practices

1. **Always use LangSmith in production** to catch issues early
2. **Tag runs by environment** (dev/staging/prod) for easy filtering
3. **Create comprehensive datasets** covering edge cases
4. **Run evaluations on every deployment** to prevent regressions
5. **Monitor trends over time** to catch quality degradation
6. **Use metadata** to track versions, users, and context

## 📖 Additional Resources

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)

## ❓ Troubleshooting

**Import errors?**
```bash
pip install -r requirements.txt
```

**LangSmith not tracing?**
- Check `.env` has `LANGSMITH_API_KEY`
- Verify project name in `LANGSMITH_PROJECT`
- Ensure environment variables are loaded

**OpenAI API errors?**
- Verify `OPENAI_API_KEY` is valid
- Check API quota and billing

## 📝 License

MIT
