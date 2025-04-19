import os

def get_token():
    # Dummy token logic for now
    return os.getenv("CANVAS_API_TOKEN", "demo-token")

def validate_login():
    token = get_token()
    return token != "demo-token"
import os

def get_headers():
    token = os.getenv("CANVAS_API_TOKEN")  # Make sure this is set in your environment
    if not token:
        raise ValueError("❌ Canvas API token not set in environment variable: CANVAS_API_TOKEN")

    return {
        "Authorization": f"Bearer {token}"
    }
