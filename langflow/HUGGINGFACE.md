# Hugging Face Spaces Deployment

This repo should be deployed as a Gradio Space.

## What Gets Deployed

Hugging Face runs:

```text
app.py
```

That Gradio app calls your Langflow API through:

```text
LANGFLOW_BASE_URL/api/v1/run/LANGFLOW_FLOW_ID
```

The Space is only the frontend. Your Langflow backend must be reachable from
the public internet.

## 1. Make Langflow Public

Do not use this in Hugging Face:

```env
LANGFLOW_BASE_URL=http://localhost:7860
```

From inside a Space, `localhost` means the Hugging Face container, not your
computer.

Use one of these instead:

```env
LANGFLOW_BASE_URL=https://your-langflow-backend.example.com
```

Good backend options:

- Langflow Cloud
- a VPS
- Railway
- Render
- Fly.io
- any HTTPS host that runs Langflow and exposes the API

Before deploying the Space, confirm this works from outside your machine:

```bash
curl https://your-langflow-backend.example.com/health
```

If your Langflow host does not expose `/health`, test the exact API URL from
Langflow's `Share -> API access` panel instead.

## 2. Create The Space

In Hugging Face:

1. Create a new Space.
2. Select `Gradio` as the SDK.
3. Use Python `3.11`.
4. Push this repository to the Space repo.

The root `README.md` already contains the Space config:

```yaml
---
title: Langflow Chatbot
sdk: gradio
app_file: app.py
python_version: 3.11
---
```

## 3. Add Space Variables And Secrets

In the Space page, open `Settings`.

Add these as variables:

```text
LANGFLOW_BASE_URL
LANGFLOW_FLOW_ID
```

Add these as secrets:

```text
LANGFLOW_API_KEY
```

Also add the provider key your flow needs as a secret:

```text
OPENAI_API_KEY
ANTHROPIC_API_KEY
GOOGLE_API_KEY
GOOGLE_GENERATIVE_AI_API_KEY
```

Only add the provider keys that your actual Langflow flow uses.

Optional variables:

```text
LANGFLOW_INPUT_TYPE=chat
LANGFLOW_OUTPUT_TYPE=chat
```

Optional secret:

```text
LANGFLOW_TWEAKS_JSON={"ComponentName":{"field":"value"}}
```

## 4. Push To Hugging Face

One common flow:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push hf main
```

If your local branch is not named `main`, push it as `main`:

```bash
git push hf HEAD:main
```

## 5. Debug The Build

If the Space builds but the chat fails:

- `Missing required environment variable`: add the missing variable or secret in Space settings.
- connection error: `LANGFLOW_BASE_URL` is not public or Langflow is down.
- `403 Forbidden`: `LANGFLOW_API_KEY` is missing, invalid, or for the wrong Langflow backend.
- provider key error: add the model provider key used by the flow as a Space secret.
- unexpected response shape: update `extract_text()` in `langflow_client.py`.

Any Space variable or secret change restarts the app automatically.
