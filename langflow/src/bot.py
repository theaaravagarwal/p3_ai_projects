#!/usr/bin/env python3
"""
Assistant - A low-cost general-purpose assistant
Name: "Assistant"
Purpose: Answer trivia, explain concepts, and provide calm, supportive conversation
Avatar: A neutral chat icon
"""

import os
import json
from src.openai_config import (
    get_max_output_tokens,
    get_openai_client,
    get_openai_model,
    get_reasoning_effort,
    get_temperature,
    has_openai_api_key,
    supports_reasoning_effort,
    trim_conversation_history,
)
from src.system_prompt import SYSTEM_PROMPT

# Initialize OpenAI client
client = get_openai_client()
API_AVAILABLE = client is not None

# Chatbot configuration
CHATBOT_CONFIG = {
    "name": "Assistant",
    "avatar": "💬",
    "description": "A low-cost general-purpose assistant",
    "version": "1.0.0"
}

def create_message_with_history(user_message: str, conversation_history: list) -> str:
    """
    Send a message to OpenAI and get a response from Assistant.
    
    Args:
        user_message: The user's input message
        conversation_history: List of previous messages in format [{"role": "user/assistant", "content": "..."}]
    
    Returns:
        The assistant's response
    """
    if not API_AVAILABLE:
        return "Error: OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
    
    # Add current user message to history
    trimmed_history = trim_conversation_history(conversation_history)
    trimmed_history.append({
        "role": "user",
        "content": user_message
    })
    
    try:
        response = client.chat.completions.create(
            model=get_openai_model(),
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }
            ] + trimmed_history,
            max_tokens=get_max_output_tokens(),
            temperature=get_temperature(),
            **(
                {"reasoning_effort": get_reasoning_effort()}
                if supports_reasoning_effort(get_openai_model())
                else {}
            ),
        )
        
        assistant_message = response.choices[0].message.content
        
        # Add assistant response to history
        trimmed_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        conversation_history[:] = trimmed_history
        return assistant_message
    
    except Exception as e:
        return f"I encountered an error: {str(e)}. Please try again."

def run_interactive_session():
    """Run an interactive chat session with Assistant."""
    print(f"\n{'='*60}")
    print(f"Welcome to {CHATBOT_CONFIG['name']} {CHATBOT_CONFIG['avatar']}")
    print(f"Your calm general-purpose assistant!")
    print(f"{'='*60}\n")
    
    conversation_history = []
    
    while True:
        print("\n(Type 'quit' to exit, 'clear' to start new conversation)\n")
        user_input = input("You: ").strip()
        
        if user_input.lower() == "quit":
            print("\nThank you for using Assistant! Take care. 💬")
            break
        
        if user_input.lower() == "clear":
            conversation_history = []
            print("\nConversation cleared! Starting fresh.\n")
            continue
        
        if not user_input:
            print("Please enter a question!")
            continue
        
        response = create_message_with_history(user_input, conversation_history)
        print(f"\n{CHATBOT_CONFIG['name']}: {response}")

if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not has_openai_api_key():
        print("\n" + "="*60)
        print("⚠️  OpenAI API Key Not Configured")
        print("="*60)
        print("\nTo use Assistant, you need to set up your OpenAI API key:")
        print("\n1. Get your API key from: https://platform.openai.com/api-keys")
        print("\n2. Create a .env file in this directory with:")
        print("   OPENAI_API_KEY=sk-your-key-here")
        print("\nOptional budget controls:")
        print("   OPENAI_MODEL=gpt-5-nano")
        print("   OPENAI_MAX_OUTPUT_TOKENS=64")
        print("   OPENAI_MAX_HISTORY_MESSAGES=2")
        print("   OPENAI_REASONING_EFFORT=minimal")
        print("\n3. Then run: uv run main.py bot")
        print("\n" + "="*60 + "\n")
        exit(1)
    
    run_interactive_session()
