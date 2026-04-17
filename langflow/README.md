# 💬 Assistant - AI Chatbot

A friendly, intelligent general-purpose chatbot for trivia, explanations, and calm conversation. Built with Langflow and OpenAI's low-cost `gpt-5-nano` by default.

## 🎯 Overview

**Name:** Assistant  
**Avatar:** 💬 (Chat bubble)  
**Purpose:** Provide neutral, helpful answers without pretending to be a specialist  
**Version:** 1.0.0

### Key Features

✅ **General Assistant** - Helps with trivia, explanations, brainstorming, and planning  
✅ **Neutral Tone** - Tries to stay objective and balance multiple viewpoints  
✅ **Safety First** - Built-in safeguards against harmful or inappropriate requests  
✅ **Deployment Ready** - Docker, API endpoints, and multiple interfaces  
✅ **Comprehensive Testing** - Tested with 7+ diverse prompts  
✅ **Conversation Memory** - Maintains context across multiple exchanges  

---

## 📋 Requirements Met

### ✅ Has a name and avatar
- **Name:** Assistant
- **Avatar:** 💬 (Chat bubble)

### ✅ Clear purpose
- Help users get concise, neutral answers
- Offer explanations, summaries, and brainstorming
- Avoid pretending to be a therapist, doctor, or other specialist

### ✅ Responds to 5+ test prompts
Test cases included:
1. Math - Algebra (linear equations)
2. Science - Biology (mitosis vs meiosis)
3. English - Essay writing (thesis statements)
4. History - Critical thinking (causes of American Revolution)
5. Study Tips - Time management strategy
6. Safety Test - Cheating prevention
7. Safety Test - Off-topic redirection

### ✅ Short system prompt with instructions
See `System Prompt` section below for full details. Key elements:
- Clear role definition
- Subject expertise areas
- Safety guidelines
- Socratic teaching approach

### ✅ Avoids unsafe/irrelevant responses
- Refuses to do homework directly
- Rejects cheating requests
- Redirects off-topic conversations
- Age-appropriate responses

### ✅ Designed for deployment
- FastAPI REST API endpoints
- Docker containerization
- Environment configuration
- Production-ready code

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
uv sync
uv run main.py setup-env
```

Then edit `.env` and add your real `OPENAI_API_KEY`.

### 2. Check status

```bash
uv run main.py status
```

### 2b. Switch model

```bash
uv run main.py model gpt-5-nano
uv run main.py model gpt-4o-mini
```

### 3. Run Tests

```bash
uv run main.py test
```

### 4. Run Interactive Session

```bash
uv run main.py bot
```

### 5. Start API Server

```bash
uv run main.py api --reload
```

### 6. Open Langflow

```bash
uv run langflow run
```

This starts the Langflow UI and seeds an `Assistant` starter flow automatically.

### One-command launcher

```bash
uv run main.py
```

This opens a small launcher menu with setup, status, bot, API, and test options.


---

## 🐳 Docker Deployment

### Option 1: Docker Compose (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up --build

# The API will be available at http://localhost:8000
```

### Option 2: Manual Docker Build

```bash
# Build the image
docker build -t assistant-homework-bot .

# Run the container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-xxx... \
  assistant-homework-bot
```

---

## 📡 API Usage

### Endpoints

#### 1. Get Chatbot Info
```bash
curl http://localhost:8000/
```

Response:
```json
{
  "name": "Assistant",
  "avatar": "💬",
  "description": "Your friendly general-purpose assistant",
  "version": "1.0.0"
}
```

#### 2. Chat Endpoint
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I solve 2x + 5 = 13?",
    "conversation_history": []
  }'
```

Response:
```json
{
  "message": "Great question! Let's work through this step by step...",
  "conversation_history": [
    {
      "role": "user",
      "content": "How do I solve 2x + 5 = 13?"
    },
    {
      "role": "assistant",
      "content": "Great question! Let's work through this step by step..."
    }
  ]
}
```

#### 3. Reset Conversation
```bash
curl -X POST http://localhost:8000/reset
```

---

## 🧪 Test Results

The chatbot has been tested with 7 diverse prompts covering:

| Test # | Category | Status | Description |
|--------|----------|--------|-------------|
| 1 | Math - Algebra | ✅ PASS | Solves equations through guided steps |
| 2 | Science - Biology | ✅ PASS | Explains mitosis vs meiosis with analogies |
| 3 | English - Essays | ✅ PASS | Guides thesis writing without writing it |
| 4 | History - Critical Thinking | ✅ PASS | Explains historical causation |
| 5 | Study Tips | ✅ PASS | Provides time management strategies |
| 6 | Safety - Cheating | ✅ PASS | Refuses direct homework completion |
| 7 | Safety - Off-topic | ✅ PASS | Redirects non-academic questions |

**Test Results:** 7/7 PASSED ✅

Run tests yourself:
```bash
uv run main.py test
```

---

## 💡 System Prompt

Assistant uses a comprehensive system prompt that:

1. **Defines Role:** Friendly general-purpose assistant for ages 14-18
2. **Teaches via Socratic Method:** Guides understanding rather than providing answers
3. **Covers Subject Expertise:**
   - Mathematics (algebra, geometry, calculus, statistics)
   - Science (physics, chemistry, biology, earth science)
   - English (literature, essays, grammar)
   - History (critical thinking, analysis)
4. **Enforces Safety:** Refuses cheating, keeps responses age-appropriate
5. **Uses Clear Language:** Analogies, numbered steps, bullet points

Example interaction showing the Socratic approach:

```
Student: "I don't understand how to factor x² + 5x + 6"
Assistant: "Great! Let's break this down. First, what two numbers 
multiply to give 6 AND add to give 5? Think about the factors of 6...
(1,6), (2,3)... which pair adds to 5?"
```

---

## 📁 Project Structure

```
langflow/
├── src/
│   ├── bot.py                   # Interactive CLI chatbot
│   └── api.py                   # FastAPI REST API server
├── tests/
│   └── test_prompts.py          # Comprehensive test suite (7+ tests)
├── pyproject.toml               # Project configuration with dependencies
├── .env.example                 # Environment variables template
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Docker Compose configuration
├── README.md                    # This file
└── .venv/                       # Virtual environment (created by uv)
```

---

## 🔧 Configuration

Edit `.env` file to customize:

```env
# Required
OPENAI_API_KEY=sk-xxx...

# Optional
API_HOST=0.0.0.0                 # API server host
API_PORT=8000                    # API server port
ENVIRONMENT=development          # development/staging/production
OPENAI_MODEL=gpt-5-nano          # Cheapest chat model this project uses by default
OPENAI_MAX_OUTPUT_TOKENS=64      # Lower this to cap response spend
OPENAI_MAX_HISTORY_MESSAGES=2    # Fewer messages = less prompt cost
OPENAI_TEMPERATURE=0.1           # Lower = more deterministic
OPENAI_REASONING_EFFORT=minimal  # Lower reasoning for less token burn
LOG_LEVEL=INFO                  # Logging level
```

---

## 📝 Usage Examples

### Interactive Mode

```bash
uv run main.py bot

# Output:
# ============================================================
# Welcome to Assistant 💬
# Your friendly general-purpose assistant!
# ============================================================
# 
# You: I don't understand quadratic equations
# Assistant: Great question! Let me help you understand quadratics...
# 
# You: quit
# Thank you for using Assistant! Keep learning! 📚
```

### API Mode

```python
import requests

response = requests.post('http://localhost:8000/chat', json={
    'message': 'How do I write a good essay introduction?',
    'conversation_history': []
})

print(response.json()['message'])
# Output: "Great question! A strong introduction has several key parts..."
```

---

## 🚢 Deployment Platforms

### Deploy to Heroku

```bash
# Create Procfile
echo "web: uvicorn src.api:app --host 0.0.0.0 --port \$PORT" > Procfile

# Deploy
heroku create assistant-homework-helper
heroku config:set OPENAI_API_KEY=sk-xxx...
git push heroku main
```

### Deploy to AWS

```bash
# Push to ECR and deploy with ECS
docker build -t assistant .
docker tag assistant:latest <AWS_ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/assistant:latest
docker push <AWS_ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/assistant:latest

# Then create ECS task and service
```

### Deploy to Azure

```bash
# Using Azure Container Instances
az container create \
  --resource-group myResourceGroup \
  --name assistant \
  --image assistant:latest \
  --ports 8000 \
  --environment-variables OPENAI_API_KEY=sk-xxx...
```

---

## 🤝 Integration Examples

### Web Frontend (React/Vue)
```javascript
async function askProfSage(question, history = []) {
  const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message: question,
      conversation_history: history
    })
  });
  return response.json();
}
```

### Slack Bot Integration
```python
from slack_sdk import WebClient

@slack_events_adapter.on("message")
def handle_messages(body, say):
    text = body["event"]["text"]
    response = requests.post('http://localhost:8000/chat', 
        json={'message': text, 'conversation_history': []})
    say(response.json()['message'])
```

---

## 🐛 Troubleshooting

**Issue:** "OPENAI_API_KEY not set"
```bash
# Solution: Create .env file
uv run main.py setup-env
# Edit .env and add your OpenAI API key
```

**Issue:** Port 8000 already in use
```bash
# Solution: Use different port
uv run main.py api --reload
```

**Issue:** Docker build fails
```bash
# Solution: Let Docker pull fresh base image
docker build --no-cache -t assistant .
```

---

## 📚 Learn More

- [Langflow Documentation](https://docs.langflow.org/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Python UV Package Manager](https://astral.sh/uv/)

---

## 📄 License

This project is created for educational purposes.

---

## 🎓 Credits

Created as a general-purpose chatbot for students to learn and understand concepts through guided tutoring by Assistant! 💬

**Features highlights:**
- 🎯 Focused on learning, not cheating
- 🔒 Safety-first design with built-in guardrails
- 🚀 Production-ready deployment
- ✅ Tested with 7+ diverse prompts
- 📱 API-first architecture

---

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Run `uv run main.py test` to verify installation
3. Check API docs at `http://localhost:8000/docs` when running

---

**Last Updated:** 2026-04-13  
**Version:** 1.0.0  
**Status:** ✅ Production Ready
