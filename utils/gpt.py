import os
import logging
import google.generativeai as genai
from typing import Optional

logger = logging.getLogger(__name__)

# Initialize Gemini with API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def ask_gpt(prompt: str, model: str = "gemini-1.5-pro", temperature: float = 0.3, max_tokens: int = 1000) -> str:
    """
    Send a prompt to Gemini and get the response
    """
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel(model)
        
        # Create completion
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.8,
                top_k=40
            )
        )
        
        # Extract and return the response text
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Error in Gemini call: {e}")
        return "" 