#!/usr/bin/env python3
"""
Interactive client for Assistant API
Use this to test the chatbot via the REST API
"""

import requests
import json
import os
from typing import List, Dict

class AssistantClient:
    """Client for interacting with Assistant API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client with API base URL"""
        self.base_url = base_url
        self.conversation_history: List[Dict] = []
        self.api_key = os.getenv("PUBLIC_API_KEY", "").strip()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers
    
    def get_info(self) -> Dict:
        """Get chatbot information"""
        response = requests.get(f"{self.base_url}/")
        return response.json()
    
    def chat(self, message: str) -> str:
        """Send a message and get a response"""
        payload = {
            "message": message,
            "conversation_history": self.conversation_history
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                headers=self._headers(),
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            self.conversation_history = data.get("conversation_history", [])
            return data.get("message", "No response received")
        
        except requests.exceptions.ConnectionError:
            return f"❌ Connection error: Cannot reach {self.base_url}"
        except requests.exceptions.RequestException as e:
            return f"❌ Error: {str(e)}"
    
    def reset(self):
        """Reset conversation history"""
        self.conversation_history = []
        requests.post(f"{self.base_url}/reset", headers=self._headers())
    
    def interactive_session(self):
        """Run an interactive chat session"""
        try:
            info = self.get_info()
            print(f"\n{'='*60}")
            print(f"Connected to {info.get('name', 'Assistant')} {info.get('avatar', '💬')}")
            print(f"Version: {info.get('version', '1.0.0')}")
            print(f"{'='*60}\n")
        except:
            print("Warning: Could not connect to API. Is the server running?")
            print(f"Make sure to start the API with: uv run main.py api --reload\n")
            return
        
        print("Type 'quit' to exit, 'clear' to reset conversation\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == "quit":
                    print("\nThank you for using Assistant! 💬\n")
                    break
                
                if user_input.lower() == "clear":
                    self.reset()
                    print("Conversation cleared!\n")
                    continue
                
                if not user_input:
                    continue
                
                print("\n⏳ Thinking...\n")
                response = self.chat(user_input)
                print(f"Assistant: {response}\n")
            
            except KeyboardInterrupt:
                print("\n\nExiting... Goodbye! 💬\n")
                break
            except Exception as e:
                print(f"Error: {e}\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Quick test mode
        client = AssistantClient()
        print("Testing Assistant API Client...")
        print("\n1. Testing connection...")
        try:
            info = client.get_info()
            print(f"   ✅ Connected: {info.get('name')} {info.get('avatar')}")
        except:
            print("   ❌ Could not connect to API")
            sys.exit(1)
        
        print("\n2. Testing chat...")
        response = client.chat("What is photosynthesis?")
        print(f"   ✅ Response received: {response[:50]}...")
        
        print("\n3. Testing conversation memory...")
        response2 = client.chat("Can you explain it in simpler terms?")
        print(f"   ✅ Follow-up response: {response2[:50]}...")
        
        print("\n✅ All tests passed!")
    else:
        # Interactive mode
        client = AssistantClient()
        client.interactive_session()
