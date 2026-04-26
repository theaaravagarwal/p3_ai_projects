# Local Deployment Guide

This project assumes a local Langflow backend and a local frontend client.

## Step 1: Get The Flow Working In Langflow

Use Langflow Playground first.

That validates:

- the prompt
- the model connection
- any tools
- memory
- vector store access

Playground is for testing inside Langflow. It is separate from your own frontend.

## Step 2: Confirm Langflow Is Running

Typical local URL:

```text
http://localhost:7860
```

Your frontend clients in this repo call Langflow at:

```text
/api/v1/run/<FLOW_ID>
```

## Step 3: Copy The Official API Settings

Inside Langflow:

1. Open the flow
2. Click `Share`
3. Click `API access`
4. Copy the Python snippet

That gives you the correct:

- server URL
- flow ID
- request shape
- auth header pattern

## Step 4: Create A Langflow API Key

Inside Langflow:

1. Open your profile
2. Go to `Settings`
3. Open `Langflow API Keys`
4. Create a key
5. Put it in `.env`

## Step 5: Configure This Repo

Create `.env` from the example:

```bash
cp .env.example .env
```

Set:

```env
LANGFLOW_BASE_URL=http://localhost:7860
LANGFLOW_FLOW_ID=your_flow_id_here
LANGFLOW_API_KEY=your_api_key_here
```

Install dependencies:

```bash
uv venv .venv
uv pip install -r requirements.txt
```

## Step 6: Test Langflow Directly

```bash
python test_langflow.py
```

Do not move on until that works.

## Step 7: Start A Frontend

### Gradio

```bash
python app.py
```

Default local URL:

```text
http://localhost:7861
```

### Streamlit

```bash
streamlit run streamlit_app.py
```

Default local URL:

```text
http://localhost:8501
```

## Who Talks To Whom

### Gradio

```text
Browser
  -> Gradio
  -> Langflow API
  -> Your flow
  -> model / tools / prompt / memory
  -> Langflow API response
  -> Gradio
  -> Browser
```

### Streamlit

```text
Browser
  -> Streamlit
  -> Langflow API
  -> Your flow
  -> Langflow API response
  -> Streamlit
  -> Browser
```

## Notes

- Langflow is the runtime and backend
- the frontend only collects user input and displays responses
- if your flow returns a different response shape, update `extract_text()` in `langflow_client.py`
