# Assistant Chatbot

Assistant is a simple OpenAI-powered chatbot with:

- a terminal chat mode
- a FastAPI server
- a small browser UI
- Docker deployment
- prompt and config tests

## What it needs

- Python 3.11+
- `uv`
- an `OPENAI_API_KEY`

## Setup

```bash
uv venv .venv
uv pip install -r requirements.txt
uv run main.py setup-env
```

Then edit `.env` and add your real `OPENAI_API_KEY`.

## Run locally

Show status:

```bash
uv run main.py status
```

Run the chatbot in the terminal:

```bash
uv run main.py bot
```

Start the API server:

```bash
uv run main.py api --reload
```

Run tests:

```bash
uv run main.py test
```

Launch the menu:

```bash
uv run main.py
```

## Configuration

The main settings live in `.env`:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_MAX_OUTPUT_TOKENS`
- `OPENAI_MAX_HISTORY_MESSAGES`
- `OPENAI_TEMPERATURE`
- `OPENAI_REQUEST_TIMEOUT_SECONDS`

## API

The server exposes:

- `GET /` for the browser UI
- `POST /chat` for sending a message
- `POST /reset` for clearing a conversation

Example request:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, can you help me?",
    "conversation_history": []
  }'
```

## Docker

Start with Docker Compose:

```bash
docker-compose up --build
```

Or use the helper script:

```bash
./docker.sh up
```

To stop:

```bash
./docker.sh down
```

## Project layout

- `main.py` - launcher and CLI entry point
- `src/api.py` - FastAPI app and browser UI
- `src/bot.py` - terminal chat loop
- `src/openai_config.py` - model and API config helpers
- `src/system_prompt.py` - chatbot behavior prompt
- `src/security.py` - API key and rate limit checks
- `tests/` - prompt and config tests

## Notes

- The app uses the current conversation history plus the system prompt.
