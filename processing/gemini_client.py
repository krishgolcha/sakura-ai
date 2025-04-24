import google.generativeai as genai
import os
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

def init_gemini():
    """Initialize Gemini API with API key"""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        logger.info("Gemini API initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini API: {e}")
        raise

def get_gemini_response(
    prompt: str,
    context: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> str:
    """Get response from Gemini API"""
    try:
        # Initialize the model
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Construct the full prompt
        full_prompt = f"Context:\n{context}\n\nQuery:\n{prompt}" if context else prompt
        
        # Generate response
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.8,
                top_k=40
            )
        )
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error getting Gemini response: {e}")
        return f"Error: Failed to get response from Gemini API - {str(e)}"

def match_course_name_gemini(query: str, courses: Dict[str, Any]) -> Dict[str, Any]:
    """Use Gemini to match course name from query"""
    try:
        # Construct prompt for course matching
        course_list = "\n".join([f"- {name}" for name in courses.keys()])
        prompt = f"""
Given the following course list:
{course_list}

And the user query:
"{query}"

Task:
1. Identify which course the user is referring to
2. If exact match found, return the course name
3. If no match found but similar courses exist, suggest them
4. If multiple matches found, list all matches
5. If no match or similar courses found, indicate that

Return your response in this format:
MATCH: <exact course name> or NONE
SIMILAR: <comma-separated similar courses> or NONE
MESSAGE: <explanation or suggestion for user>
"""
        
        response = get_gemini_response(prompt, temperature=0.3)
        
        # Parse response
        lines = response.strip().split("\n")
        result = {
            "match": None,
            "similar": [],
            "message": "No course match found"
        }
        
        for line in lines:
            if line.startswith("MATCH:"):
                match = line.replace("MATCH:", "").strip()
                if match != "NONE":
                    result["match"] = match
            elif line.startswith("SIMILAR:"):
                similar = line.replace("SIMILAR:", "").strip()
                if similar != "NONE":
                    result["similar"] = [s.strip() for s in similar.split(",")]
            elif line.startswith("MESSAGE:"):
                result["message"] = line.replace("MESSAGE:", "").strip()
        
        return result
        
    except Exception as e:
        logger.error(f"Error in course matching: {e}")
        return {
            "match": None,
            "similar": [],
            "message": f"Error matching course: {str(e)}"
        }

def process_content_gemini(content: str, query: str, tab: str) -> str:
    """Process content using Gemini API"""
    try:
        prompt = f"""
You are an AI assistant helping students find information from their Canvas course content.

Content from the {tab} tab:
{content}

User's question:
{query}

Task:
1. Find relevant information from the content
2. Provide a clear, concise answer
3. If no relevant information found, say so clearly
4. Format dates consistently like 24th March, 2025 instead of 2025-03-24.
5. For assignments, include due dates, time if mentioned, and points
6. For announcements, include post dates and time if mentioned
7. Always cite the source (tab name) where the information was found
8. Present information confidently without uncertainty
9. Use bullet points for lists
10. Include specific details like room numbers, times, and dates

Return only the relevant information in a clear, organized format with proper citations.
"""
        
        return get_gemini_response(prompt, temperature=0.5)
        
    except Exception as e:
        logger.error(f"Error processing content: {e}")
        return f"Error: Failed to process content - {str(e)}" 