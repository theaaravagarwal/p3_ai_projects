from __future__ import annotations

import os

import gradio as gr

from langflow_client import LangflowClientError, call_langflow, extract_text


def run_chat(user_message: str) -> str:
    try:
        data = call_langflow(user_message)
        return extract_text(data)
    except LangflowClientError as exc:
        return f"Langflow error: {exc}"
    except Exception as exc:
        return f"Unexpected error: {exc}"


demo = gr.Interface(
    fn=run_chat,
    inputs=gr.Textbox(lines=4, label="Ask the chatbot"),
    outputs=gr.Textbox(label="Response"),
    title="Langflow Chatbot",
    description="Gradio frontend calling a Langflow backend",
)


if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", "7861"))
    demo.launch(server_port=port)
