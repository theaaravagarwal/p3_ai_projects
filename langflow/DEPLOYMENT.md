# Deployment Guide for Assistant

## Local Development

### 1. Setup
```bash
# Navigate to project
cd /Users/aa/Documents/coding/p3ai/langflow

# Setup slim local environment with Langflow support
uv venv .venv
uv pip install -r requirements.txt
uv run main.py setup-env
# Edit .env with your OpenAI API key
```

### 2. Test the Chatbot
```bash
# Run comprehensive tests
uv run main.py test

# Or run test mode on API client
uv run client.py --test
```

### 3. Run Interactively
```bash
# Terminal-based chat
uv run main.py bot

# Or use API client
uv run client.py
```

### 4. Start API Server
```bash
# Start FastAPI server
uv run main.py api --reload

# Visit http://localhost:8000/docs for API documentation
```

---

## Docker Deployment

Docker uses the lean deployment image. It is the easiest way to run the API in
a container, but it does not install Langflow.

The Docker path is tuned for low token usage:
- `OPENAI_MODEL=gpt-5-nano`
- `OPENAI_MAX_OUTPUT_TOKENS=64`
- `OPENAI_MAX_HISTORY_MESSAGES=2`
- `OPENAI_REASONING_EFFORT=minimal`
- `OPENAI_TEMPERATURE` is ignored for GPT-5 models because they only accept the default value
- `OPENAI_REQUEST_TIMEOUT_SECONDS=45` for fail-fast request handling

For an exposed-IP deployment, set a shared API key and rate limits:
```bash
PUBLIC_API_KEY=change-me
API_RATE_LIMIT_PER_MINUTE=30
API_RATE_LIMIT_BURST=10
CORS_ORIGINS=https://your-frontend.example
TRUST_PROXY_HEADERS=true
```

Then start the public container with:
```bash
OPENAI_API_KEY=... PUBLIC_API_KEY=... ./docker.sh public-up
```

### Local Docker Testing
```bash
# Build the image
docker build -t assistant .

# Run the container
docker run -e OPENAI_API_KEY=sk-xxx... -e PUBLIC_API_KEY=change-me -p 8000:8000 assistant

# Test with docker-compose
docker-compose up --build
```

### Push to Docker Registry
```bash
# Tag for Docker Hub
docker tag assistant:latest yourusername/assistant:latest

# Push to Docker Hub
docker push yourusername/assistant:latest

# Or use AWS ECR
docker tag assistant:latest <account>.dkr.ecr.<region>.amazonaws.com/assistant:latest
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker push <account>.dkr.ecr.<region>.amazonaws.com/assistant:latest
```

---

## Cloud Platform Deployments

### Railway.app (Recommended for Beginners)
```bash
# 1. Push to GitHub
git add .
git commit -m "Initial commit: Assistant chatbot"
git push origin main

# 2. Connect repository to Railway.app
# Visit https://railway.app
# Create new project → GitHub repo → Deploy

# 3. Add environment variables in Railway dashboard
# OPENAI_API_KEY=sk-xxx...

# App will be live at: https://assistant-xxxx.railway.app
```

### Heroku
```bash
# 1. Install Heroku CLI
curl https://cli.heroku.com/install.macos.sh | sh

# 2. Create app
heroku login
heroku create assistant-homework-helper
heroku buildpacks:add heroku/python

# 3. Set environment variable
heroku config:set OPENAI_API_KEY=sk-xxx...

# 4. Deploy
git push heroku main

# Visit: https://assistant-homework-helper.herokuapp.com
```

### AWS (ECS + Fargate)
```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name assistant

# 2. Build and push image
$(aws ecr get-login --no-include-email --region us-east-1)
docker build -t assistant .
docker tag assistant:latest <account>.dkr.ecr.us-east-1.amazonaws.com/assistant:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/assistant:latest

# 3. Create ECS task definition (see ecs-task-definition.json)
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# 4. Create and run ECS service
aws ecs create-service \
  --cluster assistant-cluster \
  --service-name assistant-service \
  --task-definition assistant:1 \
  --desired-count 1 \
  --load-balancers targetGroupArn=arn:aws:elasticloadbalancing:...

# Service will be available via ALB
```

### Google Cloud Run
```bash
# 1. Configure gcloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/assistant

# 3. Deploy
gcloud run deploy assistant \
  --image gcr.io/YOUR_PROJECT_ID/assistant \
  --platform managed \
  --region us-central1 \
  --set-env-vars OPENAI_API_KEY=sk-xxx...

# Service will be available at: https://assistant-xxxx.a.run.app
```

### Azure Container Instances
```bash
# 1. Create resource group
az group create --name assistant-rg --location eastus

# 2. Create and deploy container
az container create \
  --resource-group assistant-rg \
  --name assistant \
  --image assistant:latest \
  --ports 8000 \
  --environment-variables OPENAI_API_KEY=sk-xxx... \
  --cpu 1 --memory 1

# Get public IP
az container show --resource-group assistant-rg --name assistant \
  --query ipAddress.ip --output table
```

---

## API Integration

### cURL Examples
```bash
# Get info
curl http://localhost:8000/

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain photosynthesis",
    "conversation_history": []
  }'

# Reset
curl -X POST http://localhost:8000/reset
```

### Python Integration
```python
import requests

BASE_URL = "http://localhost:8000"

# Single message
response = requests.post(f"{BASE_URL}/chat", json={
    "message": "What is calculus?",
    "conversation_history": []
})
print(response.json()['message'])

# Multi-turn conversation
history = []
for message in ["Explain derivatives", "How do I use this?", "Show an example"]:
    response = requests.post(f"{BASE_URL}/chat", json={
        "message": message,
        "conversation_history": history
    })
    data = response.json()
    history = data['conversation_history']
    print(data['message'])
```

### JavaScript/Node Integration
```javascript
const BASE_URL = "http://localhost:8000";

async function chat(message, history = []) {
  const response = await fetch(`${BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, conversation_history: history })
  });
  return response.json();
}

// Usage
const reply = await chat("What is photosynthesis?");
console.log(reply.message);
```

---

## Monitoring & Logging

### Local Logging
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
uv run main.py api --reload

# View logs
tail -f logs/app.log
```

### Production Monitoring
- **Railway:** Built-in logs at railway.app dashboard
- **Heroku:** `heroku logs --tail`
- **AWS:** CloudWatch logs in AWS console
- **Google Cloud:** `gcloud logging read "resource.type=cloud_run_revision"`
- **Azure:** Container logs in Azure portal

---

## Scaling Considerations

### Load Balancing
- All platforms support horizontal scaling
- API is stateless (uses conversation history in client)
- Can run multiple replicas

### Cost Optimization
- Use spot instances/preemptible VMs for dev
- Enable autoscaling for traffic spikes
- Consider API quotas/rate limiting

### Performance
- Response time: ~2-5 seconds (depends on OpenAI API)
- Can handle 100+ concurrent requests
- Consider caching for common questions

---

## Troubleshooting Deployment

### Port Issues
```bash
# If port 8000 is busy
uv run main.py api --reload
```

### Environment Variables
```bash
# Make sure .env or environment variables are set
export OPENAI_API_KEY=sk-xxx...
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY assistant
```

### API Key Issues
- Verify API key at https://platform.openai.com/api-keys
- Make sure key has sufficient quota
- Check usage at platform.openai.com/account/billing/overview

### Connection Issues
- Check firewall settings
- Verify port is exposed
- Test with `curl http://localhost:8000/`

---

## Security Best Practices

1. **Never commit API keys**
   - Use `.env` files (in `.gitignore`)
   - Use platform environment variables

2. **Use HTTPS in production**
   - All cloud platforms provide free HTTPS
   - Set X-Forwarded-Proto header

3. **Rate limiting** (optional)
   - Add slowapi to limit requests
   - Consider API gateway throttling

4. **Input validation**
   - Already implemented in FastAPI models
   - Messages validated for length

---

## Health Checks

### Docker/Container Health
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1
```

### Application Health
```bash
curl http://localhost:8000/
# Returns 200 if healthy
```

---

For production deployments, we recommend Railway.app or Google Cloud Run for ease of use.
