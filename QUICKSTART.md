# Quick Start Guide

Get started with LangSmith observability in 5 minutes!

## Setup (2 minutes)

### 1. Install Dependencies

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt
```

### 2. Get API Keys

**OpenAI:**
1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy the key

**LangSmith:**
1. Go to https://smith.langchain.com/
2. Sign up (free tier available)
3. Go to Settings → API Keys
4. Create and copy your API key

### 3. Configure Environment

```bash
# Copy the example file
cp .env.example .env

# Edit .env and paste your keys:
# OPENAI_API_KEY=sk-...
# LANGSMITH_API_KEY=lsv2_...
# LANGSMITH_PROJECT=llm-observability-demo
```

## Run Your First Example (3 minutes)

### Stage 1: Basic Chatbot

```bash
python step1_basic_chatbot.py
```

This runs a simple chatbot with no observability. Type messages and get responses!

### Stage 2: Prompt Templates

```bash
python step2_prompt_template.py
```

Learn how to use prompt templates and chains. See how variables work in prompts!

### Stage 3: Enable Tracing

```bash
python step3_langsmith_tracing.py
```

Now every interaction is traced! Go to https://smith.langchain.com/ to see:
- All your LLM calls
- Input prompts
- Output responses
- Token usage
- Latency

### Stage 7: Full Features

```bash
python step7_full_chatbot.py
```

Complete chatbot with:
- ✅ Automatic tracing
- ✅ Custom metadata
- ✅ LLM evaluation
- ✅ Full observability

## View Your Traces

1. Open https://smith.langchain.com/
2. Select your project (`llm-observability-demo`)
3. Browse all your traces
4. Click on any trace to see details
5. Compare runs and analyze performance

## Next Steps

Work through all 7 stages to learn:
- Stage 1: Basic chatbot
- Stage 2: Prompt templates
- Stage 3: LangSmith tracing
- Stage 4: Metadata & tags
- Stage 5: LLM-as-a-judge evaluation
- Stage 6: Dataset testing
- Stage 7: Complete integration

Read the full [README.md](README.md) for detailed explanations!

## Troubleshooting

**"Module not found" errors?**
```bash
pip install -r requirements.txt
```

**Not seeing traces in LangSmith?**
- Check your `.env` file has the correct `LANGSMITH_API_KEY`
- Verify `LANGSMITH_PROJECT` is set
- Make sure you're running stage 3 or later

**OpenAI errors?**
- Verify your `OPENAI_API_KEY` is correct
- Check you have API credits
- Ensure your key is active

## Resources

- **LangSmith:** https://smith.langchain.com/
- **LangSmith Docs:** https://docs.smith.langchain.com/
- **LangChain Docs:** https://python.langchain.com/
- **OpenAI API:** https://platform.openai.com/docs

Happy tracing! 🚀
