from __future__ import annotations

import re

import streamlit as st

from langflow_client import LangflowClientError, call_langflow, extract_text


LATEX_SIGNALS = (
    "\\begin",
    "\\frac",
    "\\sum",
    "\\int",
    "\\lim",
    "\\cdot",
    "\\quad",
    "\\circ",
    "\\times",
    "\\left",
    "\\right",
)


def is_latex_like(text: str) -> bool:
    return any(signal in text for signal in LATEX_SIGNALS)


def normalize_latex(text: str) -> str:
    return re.sub(r"(?<!\\)\\\[([0-9]+pt)\]", r"\\\\[\1]", text)


def find_unescaped_closing_bracket(text: str, start: int) -> int:
    index = start
    while index < len(text):
        if text[index] == "\\" and text[index + 1 : index + 2] == "[":
            row_spacing_end = text.find("]", index + 2)
            if row_spacing_end != -1:
                index = row_spacing_end + 1
                continue

        if text[index] == "]":
            return index

        index += 1

    return -1


def render_chunk(chunk: str, *, latex: bool = False) -> None:
    chunk = chunk.strip()
    if not chunk:
        return

    if latex:
        st.latex(normalize_latex(chunk))
        return

    st.markdown(chunk)


def render_markdown_message(text: str) -> None:
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")

    cursor = 0
    while cursor < len(text):
        dollar_start = text.find("$$", cursor)
        escaped_bracket_start = text.find(r"\[", cursor)
        bracket_start = text.find("[", cursor)

        candidates = [
            position
            for position in (dollar_start, escaped_bracket_start, bracket_start)
            if position != -1
        ]
        if not candidates:
            render_chunk(text[cursor:])
            break

        start = min(candidates)
        render_chunk(text[cursor:start])

        if start == dollar_start:
            end = text.find("$$", start + 2)
            if end == -1:
                render_chunk(text[start:])
                break
            render_chunk(text[start + 2 : end], latex=True)
            cursor = end + 2
            continue

        if start == escaped_bracket_start:
            end = text.find(r"\]", start + 2)
            if end == -1:
                render_chunk(text[start:])
                break
            render_chunk(text[start + 2 : end], latex=True)
            cursor = end + 2
            continue

        if start > 0 and text[start - 1] == "\\":
            render_chunk(text[start : start + 1])
            cursor = start + 1
            continue

        end = find_unescaped_closing_bracket(text, start + 1)
        if end == -1:
            render_chunk(text[start:])
            break

        candidate = text[start + 1 : end].strip()
        if is_latex_like(candidate):
            render_chunk(candidate, latex=True)
        else:
            render_chunk(text[start : end + 1])
        cursor = end + 1


st.set_page_config(page_title="Langflow Chatbot", page_icon="💬")
st.title("Langflow Chatbot")
st.write("Ask a question below.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_markdown_message(message["text"])

user_message = st.chat_input("Type your question here")

if user_message:
    st.session_state.messages.append({"role": "user", "text": user_message})
    with st.chat_message("user"):
        render_markdown_message(user_message)

    try:
        data = call_langflow(user_message)
        bot_reply = extract_text(data)
    except LangflowClientError as exc:
        bot_reply = f"Langflow error: {exc}"
    except Exception as exc:
        bot_reply = f"Error: {exc}"

    st.session_state.messages.append({"role": "assistant", "text": bot_reply})
    with st.chat_message("assistant"):
        render_markdown_message(bot_reply)
