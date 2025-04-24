# ask.py

import argparse
import traceback
import os
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import logging
from qa_engine_rag import answer_question_rag
from canvas_api.fetch_course_data import get_all_courses, fetch_assignments, parse_canvas_date
from utils.logger import log_event
from canvas_api.fetch_section_data import (
    get_section_content,
    get_section_data,
    embed_course_content,
    ensure_embedding_dirs,
    fetch_course_users,
    get_available_tabs
)
from canvas_api.auth import get_headers
import requests
import google.generativeai as genai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import pytz
from functools import lru_cache
from cachetools import TTLCache
from utils.cache import get_cached_content
from processing.gemini_client import init_gemini, get_gemini_response
from dateutil import parser

# Configure logging
logging.basicConfig(
    level=logging.ERROR,  # Only show errors
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Silence all loggers
logging.getLogger().setLevel(logging.ERROR)
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Initialize Gemini silently
init_gemini()

CONFIG_FILE = "config.json"
CACHE_TTL = 3600  # Cache responses for 1 hour
CACHE_MAX_SIZE = 1000

# Initialize caches
response_cache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL)
embedding_cache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL * 24)  # Cache embeddings for 24 hours

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Add embedding cache
@lru_cache(maxsize=1000)
def get_embedding(text: str) -> np.ndarray:
    """Get embedding for text with two-level caching."""
    # Check TTL cache first
    cache_key = hash(text)
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    
    # Generate new embedding
    embedding = model.encode([text])[0]
    embedding_cache[cache_key] = embedding
    return embedding

def load_config() -> dict:
    """Load configuration from config file or environment variables."""
    config = {
        "canvas_api_token": os.getenv("CANVAS_API_TOKEN"),
        "canvas_api_url": os.getenv("CANVAS_API_URL", "https://canvas.illinois.edu/api/v1"),
        "google_api_key": os.getenv("GOOGLE_API_KEY")
    }
    
    # Try loading from config file if exists
    config_path = Path(CONFIG_FILE)
    if config_path.exists():
        try:
            with open(config_path) as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            print(f"Error loading config file: {e}")
    
    return config

def check_setup() -> tuple[bool, str]:
    """Check if the system is properly configured."""
    config = load_config()
    
    missing = []
    if not config.get("canvas_api_token"):
        missing.append("CANVAS_API_TOKEN")
    if not config.get("google_api_key"):
        missing.append("GOOGLE_API_KEY")
    
    if missing:
        return False, f"Missing required configuration: {', '.join(missing)}"
    
    return True, "Setup OK"

def list_courses():
    """List all available courses."""
    try:
        courses = get_all_courses()
        if not courses:
            print("\nNo courses found. Please check your Canvas API token.")
            return
        
        print("\nAvailable courses:")
        print("-" * 80)
        for course in courses:
            if isinstance(course, dict) and 'name' in course:
                print(f"• {course['name']} (ID: {course['id']})")
        print("-" * 80)
        return
        
    except Exception as e:
        print(f"\nError listing courses: {str(e)}")
        return

def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Ask questions about your Canvas courses using natural language.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What are the office hours for CS 225?"
  %(prog)s "When is the next assignment due in IS 327?"
  %(prog)s --list-courses
  %(prog)s --debug "What is the grading policy?"

Environment Variables:
  CANVAS_API_TOKEN    Your Canvas API token
  GOOGLE_API_KEY      Your Google API key for Gemini
  CANVAS_API_URL      Canvas API URL (optional)

Configuration:
  You can also create a config.json file with the following structure:
  {
      "canvas_api_token": "your_token",
      "google_api_key": "your_key",
      "canvas_api_url": "your_url"  # optional
  }
"""
    )
    
    parser.add_argument(
        "question",
        type=str,
        nargs="*",
        help="Your question about a Canvas course"
    )
    
    parser.add_argument(
        "--list-courses",
        action="store_true",
        help="List all available courses"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser

def format_points(points: Any) -> str:
    """Format points to remove .0 and handle None/not specified cases"""
    if points is None or points == 'Not specified':
        return 'Not specified'
    try:
        num = float(points)
        return str(int(num)) if num.is_integer() else f"{num:.1f}"
    except (ValueError, TypeError):
        return str(points)

def format_date(date_obj: Optional[datetime]) -> str:
    """Format date in Central time"""
    if not date_obj:
        return "No due date set"
    central = pytz.timezone('America/Chicago')
    central_time = date_obj.astimezone(central)
    return central_time.strftime("%B %d, %Y at %I:%M %p %Z")

def get_course_id(course_name: str) -> Optional[int]:
    """Get course ID from course name using Gemini with improved matching"""
    try:
        courses = get_all_courses()
        if not courses:
            return None
            
        # First try: Direct matching for course codes
        course_name_lower = course_name.lower().replace(" ", "")
        for course in courses:
            if not isinstance(course, dict) or 'name' not in course:
                continue
                
            # Try matching course code (e.g., "IS204" or "IS 204")
            course_name_parts = course['name'].split('-')
            if len(course_name_parts) > 1:
                course_code = ''.join(filter(lambda x: x.isalnum(), course_name_parts[1])).lower()
                if course_code in course_name_lower or course_name_lower in course_code:
                    return course['id']
            
            # Also try matching against the full course name
            course_name_clean = ''.join(filter(lambda x: x.isalnum(), course['name'])).lower()
            if course_name_lower in course_name_clean:
                return course['id']
        
        # Second try: Use Gemini for semantic matching
        prompt = f"""
Given these courses:
{json.dumps(courses, indent=2)}

And this query: {course_name}

Task:
1. Find the best matching course ID
2. Consider these matching rules:
   - Exact matches for course codes (e.g., "IS 204" matches "IS204")
   - Course names containing the search terms
   - Current semester courses preferred over past ones
3. Return ONLY the course ID number, nothing else
4. If no match found, return "None"
"""
        
        result = get_gemini_response(prompt, temperature=0.3)
        
        if result and result.strip().isdigit():
            course_id = int(result.strip())
            # Verify the match
            for course in courses:
                if course.get("id") == course_id:
                    return course_id
        
        return None
    except Exception as e:
        print(f"Error in get_course_id: {str(e)}")
        return None

def get_relevant_tabs(course_id: int, question: str) -> List[str]:
    """Get relevant tabs for a question using Gemini with enhanced fallback mechanisms"""
    try:
        # Get available tabs
        tabs = list(get_section_data(str(course_id)).keys())
        
        # Use Gemini for semantic matching with detailed tab descriptions
        prompt = f"""
Given these available tabs: {', '.join(tabs)}
And this question: "{question}"

Here's what each tab typically contains:

PEOPLE:
- Course staff details (names, roles, pronouns)
- Student roster and enrollment information
- Teaching team information
- Groups

SYLLABUS:
- Course policies and requirements
- Office hours and contact information
- Course schedule and important dates
- Grading policies
- Course objectives and expectations

HOME:
- Course overview and introduction
- Instructor and TA information, office hours
- Important announcements
- Quick access to course staff information
- General course information

ANNOUNCEMENTS:
- Important updates and notices
- Time-sensitive information
- Course changes or modifications
- Upcoming deadlines
- Communication from instructors

MODULES:
- Course content organized by week/topic
- Files and resources
- Assignments (sometimes)
- Zoom links (sometimes)

ASSIGNMENTS:
- Homework and project details
- Due dates and submission instructions
- Assignment requirements
- Grading rubrics
- Submission status

FILES:
- Course documents and materials
- Lecture slides and notes
- Reading materials
- Supplementary resources
- Downloadable content

DISCUSSIONS:
- Course forums and conversations
- Student-instructor communication
- Group discussions
- Topic-based conversations
- Class participation

For a question about "{question}", I need EXACTLY three tabs that would be most relevant, in order of importance.
Consider that contact information and office hours might be found in multiple places.
You must return exactly 3 tab names from {', '.join(tabs)}, separated by commas.
Return ONLY the tab names, no other text.
"""
        response = get_gemini_response(prompt, temperature=0.1)
        
        # Clean and parse the response
        cleaned_response = response.strip().replace('\n', '').replace('.', '')
        ranked_tabs = [tab.strip() for tab in cleaned_response.split(',')]
        
        # Filter out any invalid tabs
        valid_ranked_tabs = [tab for tab in ranked_tabs if tab in tabs]
        
        # If we don't have enough valid tabs, add default ones
        if len(valid_ranked_tabs) < 3:
            default_order = ["People", "Syllabus", "Home", "Announcements"]
            for tab in default_order:
                if tab in tabs and tab not in valid_ranked_tabs:
                    valid_ranked_tabs.append(tab)
                    if len(valid_ranked_tabs) == 3:
                        break
        
        # Ensure we return exactly 3 tabs if possible
        result = valid_ranked_tabs[:3]
        print(f"\nRanked tabs: {', '.join(result)}")
        
        if len(result) < 3 and len(tabs) >= 3:
            # Add any remaining tabs to reach 3
            remaining_tabs = [tab for tab in tabs if tab not in result]
            result.extend(remaining_tabs[:3 - len(result)])
        
        return result
        
    except Exception as e:
        print(f"Error in get_relevant_tabs: {str(e)}")
        # Return at least People tab plus any other available tabs
        default_result = ["People"] if "People" in tabs else []
        other_tabs = [tab for tab in tabs if tab != "People"]
        default_result.extend(other_tabs[:2])
        return default_result

def search_section_content(content: str, question: str) -> str:
    """Search section content using FAISS"""
    try:
        # Generate embeddings with caching
        question_embedding = get_embedding(question)
        
        # Increase chunk size from 512 to 1024 for faster processing
        content_chunks = [content[i:i+1024] for i in range(0, len(content), 1024)]
        
        # Batch encode chunks for better performance
        content_embeddings = model.encode(content_chunks, batch_size=32, show_progress_bar=False)
        
        # Create and search index - use IndexIVFFlat for better performance
        dimension = content_embeddings.shape[1]
        nlist = min(4, len(content_chunks))  # number of clusters
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        if not index.is_trained:
            index.train(content_embeddings)
        index.add(content_embeddings.astype('float32'))
        
        # Search with better parameters
        index.nprobe = 2  # number of clusters to visit during search
        D, I = index.search(np.array([question_embedding]).astype('float32'), k=2)
        
        # Get relevant chunks
        relevant_chunks = [content_chunks[i] for i in I[0] if i >= 0 and i < len(content_chunks)]
        return "\n\n".join(relevant_chunks) if relevant_chunks else content
    except Exception as e:
        print(f"Failed to search content: {e}")
        return content

def clean_and_limit_content(content: str, max_chars: int = 2000) -> str:
    """Clean and limit content to a maximum number of characters while preserving meaning."""
    if isinstance(content, dict):
        content = json.dumps(content)
    elif not isinstance(content, str):
        content = str(content)
        
    # Remove extra whitespace and normalize newlines
    content = ' '.join(content.split())
    
    # If content is already under limit, return as is
    if len(content) <= max_chars:
        return content
        
    # Try to find a good breaking point
    truncated = content[:max_chars]
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n')
    
    # Break at the last complete sentence if possible
    if last_period > max_chars * 0.8:  # Only use period if it's not too far back
        return truncated[:last_period + 1]
    elif last_newline > max_chars * 0.8:  # Use newline if period isn't available
        return truncated[:last_newline]
    else:
        # If no good breaking point, break at last space
        last_space = truncated.rfind(' ')
        return truncated[:last_space] if last_space > 0 else truncated

def format_content_for_prompt(content: Dict[str, Any]) -> str:
    """Format content from different tabs into a clear structure for the prompt."""
    formatted_sections = []
    
    for tab, tab_content in content.items():
        formatted_sections.append(f"=== {tab.upper()} ===")
        if isinstance(tab_content, (list, dict)):
            formatted_sections.append(json.dumps(tab_content, indent=2))
        else:
            formatted_sections.append(str(tab_content))
            
    return "\n\n".join(formatted_sections)

def get_answer(course_id: str, question: str) -> str:
    """Get an answer to a question about a course."""
    try:
        # Get relevant tabs based on the question
        relevant_tabs = get_relevant_tabs(course_id, question)
        
        # Initialize data collection
        data_for_prompt = {
            "question": question,
            "content": {},
            "tabs_checked": relevant_tabs
        }
        
        # Get content for each relevant tab
        for tab in relevant_tabs:
            content = get_section_content(course_id, tab)
            if content:
                data_for_prompt["content"][tab] = content
        
        # Check if we have sufficient data
        if not data_for_prompt["content"]:
            return "I couldn't find relevant information to answer your question. Please try rephrasing or asking about a different topic."
        
        # Generate the final prompt
        prompt = f"""
Based on the following course information, answer this question: "{question}"

Available content from these tabs: {', '.join(data_for_prompt['tabs_checked'])}

{format_content_for_prompt(data_for_prompt['content'])}

Provide a brief, focused answer that includes:
- Main information requested
- Any directly relevant links or resources
- Source tab for each piece of information

Use bullet points and keep the response concise.
"""
        
        # Get response with lower temperature for more focused answers
        response = get_gemini_response(prompt, temperature=0.4)
        return response if response else "I apologize, but I couldn't generate a response. Please try asking your question differently."
        
    except Exception as e:
        print(f"Error in get_answer: {str(e)}")
        return "An error occurred while processing your question. Please try again."

def get_course_and_tabs(question: str, courses: List[Dict[str, Any]]) -> Tuple[Optional[int], List[str]]:
    """Get course ID and relevant tabs using Gemini with integrated decision making"""
    try:
        # First try to get course ID directly
        course_name = question.split()[0]  # Use first word as potential course name
        course_id = get_course_id(course_name)
        
        if course_id:
            # Get relevant tabs for this course
            relevant_tabs = get_relevant_tabs(course_id, question)
            return course_id, relevant_tabs[:3]
        
        # If direct match failed, try extracting course from the full question
        for word in question.split():
            if any(c.isdigit() for c in word):  # Look for words containing numbers
                course_id = get_course_id(word)
                if course_id:
                    relevant_tabs = get_relevant_tabs(course_id, question)
                    return course_id, relevant_tabs[:3]
        
        return None, []
            
    except Exception as e:
        print(f"Error in get_course_and_tabs: {str(e)}")
        return None, []

def main():
    """Main function to handle user queries"""
    try:
        parser = setup_argparse()
        args = parser.parse_args()
        
        # Handle list courses command
        if args.list_courses:
            try:
                list_courses()
                return 0
            except Exception as e:
                print(f"\nError listing courses: {str(e)}")
                return 1

        # Validate query input
        if not args.question:
            print("\nError: Please provide a query.")
            print("\nExample usage:")
            print("  python ask.py \"What assignments are due this week in CS 225?\"")
            print("  python ask.py --list-courses")
            return 1

        # Process the query
        query = " ".join(args.question).strip()
        
        if len(query) < 3:
            print("\nError: Query is too short. Please provide a more specific question.")
            return 1

        if len(query) > 500:
            print("\nError: Query is too long. Please keep your question under 500 characters.")
            return 1

        if query.lower() in ['help', 'h', '?']:
            print("\nHelp Information:")
            print("  - Ask questions about your Canvas courses in natural language")
            print("  - Use --list-courses to see available courses")
            print("\nExample queries:")
            print("  - \"What assignments are due this week in CS 225?\"")
            print("  - \"Show me the office hours for IS 327\"")
            print("  - \"What's the late submission policy?\"")
            return 0

        try:
            # Get all courses first
            courses = get_all_courses()
            if not courses:
                print("\nError: No courses found. Please check your Canvas API token.")
                return 1
            
            # Get course and tabs using Gemini
            course_id, relevant_tabs = get_course_and_tabs(query, courses)
            
            if not course_id:
                print("\nError: Could not determine which course you're asking about. Please specify a course name in your question.")
                return 1
            
            # Get the answer using the determined course_id
            answer = get_answer(str(course_id), query)
            
            if answer:
                print("\n" + "=" * 80)
                print(answer)
                print("=" * 80 + "\n")
            else:
                print("\nError: No response received. Please try again with a different query.")

        except KeyboardInterrupt:
            print("\n\nQuery cancelled by user.")
            return 130
        except Exception as e:
            print(f"\nError processing query: {str(e)}")
            return 1

        return 0

    except KeyboardInterrupt:
        print("\n\nProgram terminated by user.")
        return 130
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
