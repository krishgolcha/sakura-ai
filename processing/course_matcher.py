import json
from utils.gpt import ask_gpt
from canvas_api.fetch_course_data import get_all_courses
from utils.logger import log_event
from functools import lru_cache
import re
import logging
from processing.gemini_client import init_gemini, get_gemini_response
from typing import Optional, Dict, Any, List, Union
import google.generativeai as genai

logger = logging.getLogger(__name__)

# Initialize Gemini
init_gemini()

@lru_cache(maxsize=1)
def get_cached_courses():
    """Cache course data to avoid repeated API calls."""
    return get_all_courses()

def normalize_course_name(name: str) -> str:
    """Normalize course name for better matching."""
    # Remove spaces between course code and number
    name = re.sub(r'(\w+)\s+(\d+)', r'\1\2', name.upper())
    # Remove special characters
    return re.sub(r'[^\w\s]', '', name)

@lru_cache(maxsize=100)
def match_course_name_gpt(query: str, courses: Any = None) -> Optional[Dict]:
    """Match a course name using efficient matching and GPT-3.5-turbo as fallback."""
    if courses is None:
        courses = get_cached_courses()

    # Convert dictionary to list if needed
    if isinstance(courses, dict):
        courses = [{"id": k, "name": v} for k, v in courses.items()]
    elif not isinstance(courses, list):
        logger.error(f"Invalid courses type: {type(courses)}")
        return None

    # Normalize query
    query_norm = normalize_course_name(query)
    
    # Quick exact match first
    for course in courses:
        name = course.get('name', '')
        name_norm = normalize_course_name(name)
        
        # Check for exact matches
        if query_norm in name_norm or name_norm in query_norm:
            course["match_type"] = "exact"
            return course
            
        # Check for course code matches (e.g., "IS327" matches "IS 327")
        course_codes = re.findall(r'([A-Z]+\d+)', name_norm)
        query_codes = re.findall(r'([A-Z]+\d+)', query_norm)
        if any(qc in course_codes for qc in query_codes):
            course["match_type"] = "code"
            return course

    # If no exact match, try GPT-3.5-turbo with a simplified prompt
    course_list_str = "\n".join(f"{c['id']}: {c.get('name', 'Unnamed Course')}" for c in courses)

    prompt = f"""Match this query to one of these courses. Return ONLY the course ID number, or 'CLARIFY: [reason]' if unclear.

Query: "{query}"

Courses:
{course_list_str}"""

    response = ask_gpt(prompt, model="gpt-3.5-turbo")
    response = response.strip()

    # If it's just a number, it's a direct match
    if response.isdigit():
        for c in courses:
            if str(c["id"]) == response:
                c["match_type"] = "gpt"
                return c

    # If it starts with CLARIFY, it needs more information
    if response.startswith("CLARIFY:"):
        return {
            "match_type": "clarify",
            "message": response[9:].strip()
        }

    return None

def match_course_name(query: str, courses: Dict[str, Any]) -> Optional[str]:
    """Match course name using Gemini"""
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
        
        # Initialize model
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1000,
                top_p=0.8,
                top_k=40
            )
        )
        
        # Parse response
        lines = response.text.strip().split("\n")
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
        
        return result["match"]
        
    except Exception as e:
        logger.error(f"Error in course matching: {e}")
        return None

def get_course_id(course_name: str) -> Optional[int]:
    """Get course ID from course name"""
    try:
        courses = get_cached_courses()
        matched_course = match_course_name_gpt(course_name, courses)
        
        if matched_course and isinstance(matched_course, dict) and "id" in matched_course:
            return matched_course["id"]
            
        return None
        
    except Exception as e:
        logger.error(f"Error getting course ID: {e}")
        return None
