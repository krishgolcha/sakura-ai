import google.generativeai as genai
import os

# Initialize Gemini with API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ GOOGLE_API_KEY not found in environment variables")
    exit(1)

genai.configure(api_key=api_key)

# List available models
print("Available models:")
for model in genai.list_models():
    print(f"- {model.name}")
    print(f"  Description: {model.description}")
    print(f"  Supported methods: {model.supported_generation_methods}")
    print() 