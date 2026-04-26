from __future__ import annotations

import streamlit as st

from langflow_client import LangflowClientError, call_langflow, extract_text


st.set_page_config(page_title="Langflow Chatbot", page_icon="💬")
st.title("Langflow Chatbot")
st.write("Ask a question below.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

user_message = st.chat_input("Type your question here")

if user_message:
    st.session_state.messages.append({"role": "user", "text": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    try:
        data = call_langflow(user_message)
        bot_reply = extract_text(data)
    except LangflowClientError as exc:
        bot_reply = f"Langflow error: {exc}"
    except Exception as exc:
        bot_reply = f"Error: {exc}"

    st.session_state.messages.append({"role": "assistant", "text": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
