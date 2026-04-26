import requests
import os
import uuid

api_key = 'YOUR_API_KEY_HERE'
url = "http://localhost:7860/api/v1/run/24fffa37-3af0-4d1a-9890-7ff77a3be8cf"  # The complete API endpoint URL for this flow

# Request payload configuration
payload = {
    "output_type": "chat",
    "input_type": "chat",
    "input_value": "Hello"
}
payload["session_id"] = str(uuid.uuid4())

headers = {"x-api-key": api_key}

try:
    # Send API request
    response = requests.request("POST", url, json=payload, headers=headers)
    response.raise_for_status()  # Raise exception for bad status codes

    # Print response
    print(response.text)

except requests.exceptions.RequestException as e:
    print(f"Error making API request: {e}")
except ValueError as e:
    print(f"Error parsing response: {e}")
