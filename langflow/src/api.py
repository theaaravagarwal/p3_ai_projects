#!/usr/bin/env python3
"""
FastAPI web server for Assistant general-purpose assistant
This provides a REST API endpoint for deployment
"""

import os
import json
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

# Initialize FastAPI app
app = FastAPI(
    title="Assistant",
    description="A low-cost general-purpose assistant with supportive conversation",
    version="1.0.0"
)

# Add CORS middleware for web deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = get_openai_client()

# Data models
class Message(BaseModel):
    """A single message in the conversation"""
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    """Request body for chat endpoint"""
    message: str
    conversation_history: List[Message] = []

class ChatResponse(BaseModel):
    """Response body for chat endpoint"""
    message: str
    conversation_history: List[Message]

# Chatbot configuration
CHATBOT_CONFIG = {
    "name": "Assistant",
    "avatar": "💬",
    "description": "A low-cost general-purpose assistant",
    "version": "1.0.0"
}

@app.get("/")
async def root():
    """Root endpoint - returns chatbot info"""
    return {
        "name": CHATBOT_CONFIG["name"],
        "avatar": CHATBOT_CONFIG["avatar"],
        "description": CHATBOT_CONFIG["description"],
        "version": CHATBOT_CONFIG["version"],
        "endpoints": {
            "info": "/info",
            "chat": "/chat (POST)"
        }
    }

@app.get("/info")
async def info():
    """Get chatbot information"""
    return CHATBOT_CONFIG

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with Assistant
    
    Args:
        request: ChatRequest with message and conversation history
    
    Returns:
        ChatResponse with assistant message and updated history
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if not has_openai_api_key():
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    try:
        # Convert Pydantic models to dicts for OpenAI API
        conversation = [{"role": m.role, "content": m.content} for m in request.conversation_history]
        conversation = trim_conversation_history(conversation)
        
        # Add user message
        conversation.append({"role": "user", "content": request.message})
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=get_openai_model(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT}
            ] + conversation,
            max_tokens=get_max_output_tokens(),
            temperature=get_temperature(),
            **(
                {"reasoning_effort": get_reasoning_effort()}
                if supports_reasoning_effort(get_openai_model())
                else {}
            ),
        )
        
        assistant_message = response.choices[0].message.content
        
        # Update conversation history
        conversation.append({"role": "assistant", "content": assistant_message})
        
        # Convert back to Message objects
        history = [Message(role=m["role"], content=m["content"]) for m in conversation]
        
        return ChatResponse(
            message=assistant_message,
            conversation_history=history
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/reset")
async def reset_conversation():
    """Reset conversation history"""
    return {
        "status": "conversation reset",
        "message": "Start a new conversation with Assistant!"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Check for API key
    if not has_openai_api_key():
        print("Error: OPENAI_API_KEY environment variable not set!")
        print("Please create a .env file with: OPENAI_API_KEY=your_key_here")
        exit(1)
    
    print(f"Starting {CHATBOT_CONFIG['name']} 💬 on http://0.0.0.0:8000")
    print("API docs available at http://0.0.0.0:8000/docs")
    
    # Run with: uvicorn src.api:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
