"""
Assistant package
"""

__version__ = "1.0.0"
__author__ = "P3AI"
__description__ = "Low-cost general-purpose assistant"

from .bot import create_message_with_history, SYSTEM_PROMPT, CHATBOT_CONFIG

__all__ = [
    "create_message_with_history",
    "SYSTEM_PROMPT",
    "CHATBOT_CONFIG"
]
