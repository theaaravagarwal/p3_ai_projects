---
title: Langflow Chatbot
sdk: gradio
app_file: app.py
python_version: 3.11
---

# Langflow Local Clients

This repo is a thin local frontend layer for a Langflow flow.

Use it when:

- Langflow is already running locally
- your flow is already working in Langflow Playground
- you want a separate frontend that calls the Langflow API

## Architecture

### Playground testing

```text
Browser
  -> Langflow UI
  -> Playground
  -> Your flow
```

### Local frontend plus local Langflow backend

```text
Browser
  -> Gradio or Streamlit
  -> POST /api/v1/run/<FLOW_ID>
  -> Langflow
  -> Your flow
  -> model / tools / memory
```

Key point:

- Gradio or Streamlit is only the frontend
- Langflow stores and runs the flow
- Langflow exposes the API endpoint

## Setup

1. Start Langflow locally and confirm your flow works in Playground.
2. Copy the official API snippet from `Share -> API access` in Langflow.
3. Create a Langflow API key in `Settings -> Langflow API Keys`.
4. Copy `.env.example` to `.env` and fill in:
   - `LANGFLOW_BASE_URL`
   - `LANGFLOW_FLOW_ID`
   - `LANGFLOW_API_KEY`
   - the provider key your flow actually uses, such as `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`

Install dependencies:

```bash
uv venv .venv
uv pip install -r requirements.txt
```

If you prefer `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Test The Langflow API First

Do not move to a frontend until the direct API call works.

```bash
python test_langflow.py
```

This sends `"Hello"` to your configured flow and prints the raw JSON plus the extracted reply text.

## Run The Gradio Frontend

```bash
python app.py
```

Default URL:

- `http://localhost:7861`

Request path:

- browser -> Gradio -> Langflow API -> flow -> response -> Gradio -> browser

## Deploy On Hugging Face Spaces

Use the Gradio Space path for this repo. Hugging Face reads the YAML block at
the top of this README and starts `app.py`.

Important: the Space cannot call `http://localhost:7860` on your laptop.
Set `LANGFLOW_BASE_URL` to a public Langflow backend URL, such as Langflow
Cloud or Langflow running on a VPS, Railway, Render, Fly.io, etc.

In your Hugging Face Space, add these as Settings -> Variables or Secrets:

- Variable: `LANGFLOW_BASE_URL`
- Variable: `LANGFLOW_FLOW_ID`
- Secret: `LANGFLOW_API_KEY`
- Secret: the provider key used by your flow, such as `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
- Optional variable: `LANGFLOW_INPUT_TYPE`
- Optional variable: `LANGFLOW_OUTPUT_TYPE`
- Optional secret: `LANGFLOW_TWEAKS_JSON`

See `HUGGINGFACE.md` for the full deploy checklist.

## Run The Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

Default URL:

- `http://localhost:8501`

Request path:

- browser -> Streamlit -> Langflow API -> flow -> response -> Streamlit -> browser

## Files

- `langflow_client.py`: shared request and response parsing logic
- `test_langflow.py`: direct API connectivity check
- `app.py`: Gradio frontend
- `streamlit_app.py`: Streamlit frontend

## Troubleshooting

- `403 Forbidden`: the API key is missing, invalid, or not being sent in `x-api-key`
- connection error: Langflow is not running at `LANGFLOW_BASE_URL`
- provider credential error: set the matching provider key in `.env` such as `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. The client forwards these to Langflow as `X-LANGFLOW-GLOBAL-VAR-*` headers for each request.
- unexpected response shape: inspect the raw output from `python test_langflow.py`

If you need to override a component at request time, set `LANGFLOW_TWEAKS_JSON` to a valid Langflow `tweaks` object.

If your flow returns a slightly different JSON shape, update `extract_text()` in `langflow_client.py`.
