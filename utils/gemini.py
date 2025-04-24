import os
import google.generativeai as genai
from typing import Optional
import logging

def init_gemini():
    """Initialize the Gemini API with the API key."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        logging.error(f"Failed to initialize Gemini: {e}")
        return False

def get_gemini_response(prompt: str, temperature: float = 0.7) -> Optional[str]:
    """Get a response from Gemini API.
    
    Args:
        prompt (str): The prompt to send to Gemini
        temperature (float): Controls randomness in the response (0.0 to 1.0)
        
    Returns:
        Optional[str]: The response text or None if there's an error
    """
    try:
        # Configure the model
        model = genai.GenerativeModel('gemini-pro')
        
        # Generate the response
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                top_p=0.8,
                top_k=40,
                max_output_tokens=2048,
            )
        )
        
        # Return the response text
        return response.text if response else None
        
    except Exception as e:
        logging.error(f"Error getting Gemini response: {e}")
        return None 