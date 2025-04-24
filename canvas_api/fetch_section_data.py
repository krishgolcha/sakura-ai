import requests
import json
from bs4 import BeautifulSoup
import logging
from typing import Optional, Dict, Any, List, Tuple
from canvas_api.auth import get_headers
from canvas_api.rate_limiter import CanvasRateLimiter
import re
from datetime import datetime, timedelta
import pytz
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse as dateutil_parse
import time
from canvas_api.tab_priorities import get_tab_priority_prompt
from canvas_api.fetch_modules import get_module_content
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)
BASE_URL = "https://canvas.illinois.edu/api/v1"

# Initialize rate limiter
rate_limiter = CanvasRateLimiter(max_requests_per_minute=60)

# Add paths for embedding storage
EMBEDDINGS_DIR = Path("embeddings")
COURSE_EMBEDDINGS_DIR = EMBEDDINGS_DIR / "courses"
SECTION_EMBEDDINGS_DIR = EMBEDDINGS_DIR / "sections"
FAISS_INDEX_DIR = EMBEDDINGS_DIR / "faiss"

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cache UTC timezone object at module level for better performance
UTC_TZ = pytz.UTC

class ContentError(Exception):
    """Custom exception for content-related errors"""
    pass

class EmbeddingError(Exception):
    """Custom exception for embedding-related errors"""
    pass

def ensure_embedding_dirs():
    """Ensure embedding directories exist"""
    COURSE_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    SECTION_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

def is_course_embedded(course_id: str) -> bool:
    """Check if a course is already embedded"""
    course_file = COURSE_EMBEDDINGS_DIR / f"{course_id}.json"
    return course_file.exists()

def is_section_embedded(course_id: str, section_name: str) -> bool:
    """Check if a course section is already embedded"""
    section_file = SECTION_EMBEDDINGS_DIR / f"{course_id}_{section_name}.json"
    return section_file.exists()

def get_faiss_index(course_id: str, section_name: str) -> Optional[faiss.Index]:
    """Get or create FAISS index for a section"""
    index_file = FAISS_INDEX_DIR / f"{course_id}_{section_name}.index"
    if index_file.exists():
        return faiss.read_index(str(index_file))
    return None

def create_faiss_index(text: str, course_id: str, section_name: str) -> faiss.Index:
    """Create FAISS index for a section's content"""
    # Split text into chunks
    chunks = [text[i:i+512] for i in range(0, len(text), 512)]
    
    # Generate embeddings
    embeddings = model.encode(chunks)
    
    # Create and save index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    index_file = FAISS_INDEX_DIR / f"{course_id}_{section_name}.index"
    faiss.write_index(index, str(index_file))
    
    return index

def embed_course_content(course_id: str, section_name: str) -> None:
    """Embed course content if not already embedded"""
    try:
        # Skip embedding for announcements and assignments
        if section_name.lower() in ["announcements", "assignments"]:
            return
            
        if not is_course_embedded(course_id):
            logger.info(f"Embedding course {course_id}")
            # Get course data
            course_data = get_section_data(course_id)
            if not course_data:
                raise EmbeddingError(f"Failed to get course data for {course_id}")
            
            # Save course data
            course_file = COURSE_EMBEDDINGS_DIR / f"{course_id}.json"
            with open(course_file, 'w') as f:
                json.dump(course_data, f)
        
        if not is_section_embedded(course_id, section_name):
            logger.info(f"Embedding section {section_name} for course {course_id}")
            # Get section content
            content = get_section_content(course_id, section_name)
            if not content:
                raise EmbeddingError(f"Failed to get content for section {section_name}")
            
            # Save section content
            section_file = SECTION_EMBEDDINGS_DIR / f"{course_id}_{section_name}.json"
            with open(section_file, 'w') as f:
                json.dump({"content": content}, f)
            
            # Create FAISS index
            create_faiss_index(content, course_id, section_name)
            
    except Exception as e:
        logger.error(f"Failed to embed content: {str(e)}")
        raise EmbeddingError(f"Failed to embed content: {str(e)}")

def get_cached_content(url: str, params: Optional[Dict] = None) -> Any:
    """Cache API responses with optimized TTLs based on content type"""
    try:
        # Generate cache key
        cache_key = f"{url}:{json.dumps(params, sort_keys=True) if params else ''}"
        cache_path = Path("data/cache") / f"{hashlib.md5(cache_key.encode()).hexdigest()}.json"
        
        # Determine cache TTL based on content type
        cache_ttl = timedelta(hours=1)  # Default TTL
        
        if "announcements" in url:
            cache_ttl = timedelta(minutes=15)  # Shorter TTL for announcements
        elif "assignments" in url:
            cache_ttl = timedelta(minutes=30)  # Medium TTL for assignments
        elif "syllabus" in url:
            cache_ttl = timedelta(days=1)  # Longer TTL for syllabus
        elif "modules" in url:
            cache_ttl = timedelta(hours=4)  # Medium TTL for modules
            
        # Check cache first
        if cache_path.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if cache_age < cache_ttl:
                with open(cache_path, 'r') as f:
                    return json.load(f)
        
        # Fetch new data if cache miss or expired
        response = fetch_api_data(url, params)
        logger.info(f"Raw API response from {url}: {json.dumps(response, indent=2)}")  # Debug log
        
        # For assignments, ensure we're returning a list
        if "assignments" in url:
            response = list(response) if isinstance(response, (list, tuple)) else []
            # Log each assignment's date
            for assignment in response:
                logger.info(f"Assignment: {assignment.get('name')}, Due: {assignment.get('due_at')}")
        
        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(response, f)
            
        return response
        
    except Exception as e:
        logger.error(f"Failed to get cached content: {e}")
        raise ContentError(str(e))

def sanitize_input(text: str) -> str:
    """Sanitize input to prevent injection attacks"""
    if not isinstance(text, str):
        return ""
    # Remove potential script tags and SQL injection attempts
    text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.I | re.S)
    text = text.replace("'", "''").replace(";", "")
    return text.strip()

def parse_date(date_str: str) -> datetime:
    """Parse a date string into a datetime object using multiple formats"""
    try:
        return dateutil_parse(date_str)
    except (ValueError, TypeError):
        # Try common formats if dateutil fails
        formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%m/%d/%y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%B %d %Y",
            "%b %d %Y"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Could not parse date: {date_str}")

def parse_canvas_date(date_str: str) -> Optional[datetime]:
    """Parse a Canvas API date string into a datetime object.
    
    Supports multiple date formats:
    - ISO format with 'Z' timezone (e.g., '2024-03-20T15:30:00Z')
    - ISO format with offset (e.g., '2024-03-20T15:30:00+00:00')
    - Simple date format (e.g., '2024-03-20')
    
    Args:
        date_str: A string representing a date from the Canvas API
        
    Returns:
        datetime: A timezone-aware datetime object in UTC
        None: If the date string cannot be parsed
    """
    if not date_str:
        return None
        
    logger.debug(f"Attempting to parse date string: {date_str}")
    
    try:
        # Try parsing ISO format with 'Z' timezone
        if date_str.endswith('Z'):
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            logger.debug(f"Successfully parsed ISO-Z format date: {dt}")
            return dt.astimezone(UTC_TZ)
            
        # Try parsing ISO format with offset
        if 'T' in date_str:
            dt = datetime.fromisoformat(date_str)
            logger.debug(f"Successfully parsed ISO format date: {dt}")
            return dt.astimezone(UTC_TZ)
            
        # Try parsing simple date format
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        # Make timezone-aware
        dt = UTC_TZ.localize(dt)
        logger.debug(f"Successfully parsed simple date format: {dt}")
        return dt
        
    except ValueError as e:
        logger.error(f"Failed to parse date string '{date_str}': {str(e)}")
        return None

def clean_html(html: str) -> str:
    """Clean HTML content with improved security and formatting"""
    try:
        if not html:
            return ""
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove potentially dangerous elements
        for element in soup.find_all(['script', 'style', 'iframe', 'form', 'input', 'button']):
            element.decompose()
        
        # Clean and format text
        text = []
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div', 'span']):
            element_text = element.get_text(separator=' ', strip=True)
            if element_text:
                if element.name.startswith('h'):
                    text.append(f"\n\n{element_text}\n")
                elif element.name == 'li':
                    text.append(f"• {element_text}")
                else:
                    text.append(element_text)
        
        cleaned_text = '\n'.join(text)
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        return cleaned_text.strip()
    except Exception as e:
        logger.error(f"HTML cleaning failed: {str(e)}")
        return ""

def fetch_api_data(url: str, params: Optional[Dict] = None) -> Any:
    """Enhanced API data fetching with better error handling and rate limiting"""
    try:
        # Get headers
        headers = get_headers()
        if not headers:
            raise ContentError("Failed to get API headers")
            
        # Check rate limits before making request
        rate_limiter.wait_if_needed()
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        # Handle rate limit response
        retry_after = rate_limiter.handle_rate_limit(response)
        if retry_after is not None:
            time.sleep(retry_after)
            response = requests.get(url, headers=headers, params=params, timeout=10)
        
        response.raise_for_status()
        rate_limiter.add_request()  # Record successful request
        return response.json()
        
    except requests.exceptions.Timeout:
        logger.error(f"Request timed out: {url}")
        raise ContentError("Request timed out")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        raise ContentError(f"API request failed: {str(e)}")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {str(e)}")
        raise ContentError("Invalid API response format")
        
    except Exception as e:
        logger.error(f"Unexpected error in fetch_api_data: {str(e)}")
        raise ContentError(f"Unexpected error: {str(e)}")

def get_section_data(course_id: str) -> Dict[str, Dict[str, Any]]:
    """Get all available sections for a course with validation"""
    try:
        course_id = sanitize_input(course_id)
        if not course_id.isdigit():
            raise ContentError("Invalid course ID format")
            
        url = f"{BASE_URL}/courses/{course_id}/tabs"
        response = requests.get(url, headers=get_headers(), timeout=10)
        response.raise_for_status()
        tabs = response.json()
        
        return {tab["label"]: {
            "id": tab["id"],
            "type": tab["type"],
            "position": tab["position"],
            "hidden": tab.get("hidden", False)
        } for tab in tabs if tab.get("label")}
    except Exception as e:
        logger.error(f"Failed to fetch section data for course {course_id}: {e}")
        return {}

def parse_relative_date(date_str: str) -> Optional[Tuple[datetime, datetime]]:
    """Parse relative date strings into date ranges"""
    try:
        now = datetime.now(pytz.UTC)
        date_str = date_str.lower().strip()
        
        if "last week" in date_str:
            start = now - timedelta(days=7)
            return start, now
        elif "last month" in date_str:
            start = now - relativedelta(months=1)
            return start, now
        elif "yesterday" in date_str:
            start = now - timedelta(days=1)
            end = now
            return start, end
        elif "last" in date_str and "days" in date_str:
            try:
                days = int(''.join(filter(str.isdigit, date_str)))
                start = now - timedelta(days=days)
                return start, now
            except ValueError:
                return None
        elif "after" in date_str:
            try:
                # Extract the date after "after"
                date_part = date_str.split("after")[1].strip()
                start = parse_date(date_part)
                if start:
                    if not start.tzinfo:
                        start = pytz.UTC.localize(start)
                    return start, now + relativedelta(years=1)  # Future date
            except Exception:
                return None
        
        return None
    except Exception as e:
        logger.error(f"Failed to parse relative date: {e}")
        return None

def parse_date_range(date_str: str) -> Optional[Tuple[datetime, datetime]]:
    """Parse date range strings into start and end dates"""
    try:
        # Handle relative dates first
        relative_range = parse_relative_date(date_str)
        if relative_range:
            return relative_range
            
        # Try to parse explicit date ranges
        if " to " in date_str or " - " in date_str:
            parts = date_str.replace(" - ", " to ").split(" to ")
            if len(parts) == 2:
                start = parse_date(parts[0])
                end = parse_date(parts[1])
                if start and end:
                    # Ensure dates are timezone-aware
                    if not start.tzinfo:
                        start = pytz.UTC.localize(start.replace(hour=0, minute=0, second=0))
                    if not end.tzinfo:
                        end = pytz.UTC.localize(end.replace(hour=23, minute=59, second=59))
                    return start, end
        
        # Try to parse single date (will return range for entire day)
        single_date = parse_date(date_str)
        if single_date:
            if not single_date.tzinfo:
                single_date = pytz.UTC.localize(single_date)
            start = single_date.replace(hour=0, minute=0, second=0)
            end = single_date.replace(hour=23, minute=59, second=59)
            return start, end
            
        return None
    except Exception as e:
        logger.error(f"Failed to parse date range: {e}")
        return None

def filter_announcements_by_date(announcements: List[Dict], date_range: Optional[str] = None) -> List[Dict]:
    """Filter announcements by date range"""
    try:
        if not announcements or not date_range:
            logger.info("[DEBUG] No announcements or date range provided")
            return announcements
            
        logger.info(f"[DEBUG] Parsing date range: {date_range}")
        date_tuple = parse_date_range(date_range)
        if not date_tuple:
            logger.info("[DEBUG] Failed to parse date range")
            return announcements
            
        start_date, end_date = date_tuple
        logger.info(f"[DEBUG] Date range parsed: {start_date} to {end_date}")
        filtered = []
        
        # Convert dates to UTC for consistent comparison
        current_time = datetime.now(pytz.UTC)
        logger.info(f"[DEBUG] Current time: {current_time}")
        
        for announcement in announcements:
            try:
                # Get the posted_at date from the formatted string
                posted_at_str = announcement.get("posted_at")
                logger.info(f"[DEBUG] Processing announcement: {announcement.get('title')} with posted_at: {posted_at_str}")
                
                if not posted_at_str:
                    logger.info("[DEBUG] No posted_at date found")
                    continue
                
                # Parse the date string back to datetime
                try:
                    posted_at = datetime.strptime(posted_at_str, "%Y-%m-%d %H:%M:%S UTC")
                    posted_at = pytz.UTC.localize(posted_at)
                    logger.info(f"[DEBUG] Parsed posted_at date: {posted_at}")
                except ValueError:
                    logger.info("[DEBUG] Failed to parse posted_at date")
                    continue
                
                # For relative dates (like "last week"), calculate based on current time
                if "last" in date_range.lower():
                    logger.info("[DEBUG] Using relative date comparison")
                    if start_date <= posted_at <= current_time:
                        logger.info("[DEBUG] Announcement within relative date range")
                        filtered.append(announcement)
                    else:
                        logger.info("[DEBUG] Announcement outside relative date range")
                else:
                    # For specific date ranges, use the exact range
                    logger.info("[DEBUG] Using exact date range comparison")
                    if start_date <= posted_at <= end_date:
                        logger.info("[DEBUG] Announcement within date range")
                        filtered.append(announcement)
                    else:
                        logger.info(f"[DEBUG] Announcement outside date range: {posted_at} not between {start_date} and {end_date}")
                        
            except Exception as e:
                logger.warning(f"Failed to process announcement date: {e}")
                continue
                
        logger.info(f"[DEBUG] Found {len(filtered)} announcements within date range")
        return filtered
    except Exception as e:
        logger.error(f"Failed to filter announcements by date: {e}")
        return announcements

def fetch_announcements(course_id: str, date_range: Optional[str] = None) -> str:
    """Fetch announcements for a course"""
    try:
        url = f"{BASE_URL}/courses/{course_id}/discussion_topics"
        params = {
            "only_announcements": True,
            "per_page": 50,
            "include": ["author", "attachments"]
        }
        
        try:
            logger.info(f"[DEBUG] Fetching announcements from URL: {url}")
            response = requests.get(url, headers=get_headers(), params=params, timeout=10)
            response.raise_for_status()
            announcements = response.json()
            logger.info(f"[DEBUG] Raw announcements response: {announcements}")

            # Format announcements into structured data
            formatted_announcements = []
            for announcement in announcements:
                # Extract dates with fallback options
                posted_at = None
                date_sources = [
                    announcement.get("posted_at"),
                    announcement.get("created_at"),
                    announcement.get("updated_at")
                ]
                
                logger.info(f"[DEBUG] Processing announcement: {announcement.get('title')} with date sources: {date_sources}")
                
                for date_source in date_sources:
                    posted_at = parse_date(date_source)
                    if posted_at:
                        logger.info(f"[DEBUG] Found valid date: {posted_at} from source: {date_source}")
                        break
                
                posted_str = posted_at.strftime("%Y-%m-%d %H:%M:%S UTC") if posted_at else "[Date Unknown]"
                
                # Extract message content and clean it
                message = clean_html(announcement.get("message", ""))
                
                # Format the announcement
                formatted = {
                    "title": announcement.get("title", "[Title Unknown]"),
                    "posted_at": posted_str,
                    "author": announcement.get("author", {}).get("display_name", "[Author Unknown]"),
                    "message": message,
                    "attachments": [
                        {
                            "filename": att.get("display_name", ""),
                            "url": att.get("url", "")
                        }
                        for att in announcement.get("attachments", [])
                    ]
                }
                
                formatted_announcements.append(formatted)
                logger.info(f"[DEBUG] Formatted announcement: {formatted}")
            
            # Sort announcements by date (if available)
            formatted_announcements.sort(
                key=lambda x: parse_date(x["posted_at"]) or datetime.min.replace(tzinfo=pytz.UTC),
                reverse=True
            )
            
            # Filter by date range if specified
            if date_range:
                logger.info(f"[DEBUG] Filtering announcements by date range: {date_range}")
                formatted_announcements = filter_announcements_by_date(formatted_announcements, date_range)
                logger.info(f"[DEBUG] Filtered announcements: {formatted_announcements}")
            
            return json.dumps(formatted_announcements)

        except Exception as e:
            logger.error(f"Failed to fetch announcements: {e}")
            return json.dumps({"error": str(e)})

    except Exception as e:
        logger.error(f"Failed to fetch announcements: {e}")
        return json.dumps({"error": str(e)})

def extract_dates_from_text(text: str) -> List[str]:
    """Extract dates from text content using regex patterns"""
    date_patterns = [
        r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY or M/D/YY
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?,? \d{4}\b',  # Month DD, YYYY
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?\b'  # Month DD
    ]
    
    found_dates = []
    for pattern in date_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            date_str = match.group()
            try:
                # Try to parse and standardize the date format
                parsed_date = parse_date(date_str)
                formatted_date = parsed_date.strftime("%Y-%m-%d")
                found_dates.append(formatted_date)
            except (ValueError, TypeError):
                # If parsing fails, add the original string
                found_dates.append(date_str)
    
    return sorted(list(set(found_dates)))

def fetch_course_users(course_id: str) -> List[Dict[str, Any]]:
    """
    Fetch comprehensive user data for a course including all enrollment types.
    
    Args:
        course_id: The Canvas course ID
        
    Returns:
        List of dictionaries containing user information with role-specific details
    """
    try:
        url = f"{BASE_URL}/courses/{course_id}/users"
        params = {
            "include[]": ["email", "enrollments", "avatar_url", "bio", "pronouns"],
            "enrollment_type[]": ["teacher", "ta", "designer", "student", "observer"],
            "per_page": 100  # Maximum allowed by Canvas API
        }
        
        # Use get_cached_content for better caching
        users = get_cached_content(url, params)
        if not users or not isinstance(users, list):
            logger.warning(f"No users found or invalid response format for course {course_id}")
            return []
            
        # Define role priorities and permissions
        role_priority = {
            "TeacherEnrollment": 1,
            "TaEnrollment": 2,
            "DesignerEnrollment": 3,
            "StudentEnrollment": 4,
            "ObserverEnrollment": 5
        }
        
        # Define role display names
        role_display_names = {
            "TeacherEnrollment": "Instructor",
            "TaEnrollment": "Teaching Assistant",
            "DesignerEnrollment": "Designer",
            "StudentEnrollment": "Student",
            "ObserverEnrollment": "Observer"
        }
        
        formatted_users = []
        for user in users:
            if not isinstance(user, dict):
                continue
                
            # Get all enrollments for the user
            enrollments = user.get("enrollments", [])
            if not enrollments:
                continue
                
            # Get the highest priority enrollment
            sorted_enrollments = sorted(
                enrollments,
                key=lambda x: role_priority.get(x.get("type", ""), 99)
            )
            primary_enrollment = sorted_enrollments[0]
            role = primary_enrollment.get("type", "")
            
            # Set default permissions based on role
            role_permissions = {
                "can": {
                    "view_email": role in ["TeacherEnrollment", "TaEnrollment"],
                    "grade": role in ["TeacherEnrollment", "TaEnrollment"],
                    "update_content": role in ["TeacherEnrollment", "TaEnrollment", "DesignerEnrollment"],
                    "view_grades": role in ["TeacherEnrollment", "TaEnrollment", "StudentEnrollment"],
                    "manage_content": role in ["TeacherEnrollment", "DesignerEnrollment"],
                    "manage_course": role == "TeacherEnrollment"
                }
            }
            
            formatted_user = {
                "name": user.get("name", ""),
                "email": user.get("email", ""),
                "role": role_display_names.get(role, role.replace("Enrollment", "").strip()),
                "pronouns": user.get("pronouns", ""),
                "avatar_url": user.get("avatar_url", ""),
                "bio": user.get("bio", ""),
                "section": primary_enrollment.get("course_section_id", ""),
                "status": primary_enrollment.get("enrollment_state", ""),
                "permissions": role_permissions
            }
            formatted_users.append(formatted_user)
            
        # Sort users by role priority
        formatted_users.sort(
            key=lambda x: role_priority.get(
                next((k for k, v in role_display_names.items() if v == x["role"]), ""),
                99
            )
        )
        
        logger.info(f"Found {len(formatted_users)} users in course {course_id}")
        return formatted_users
        
    except Exception as e:
        logger.error(f"Failed to fetch users for course {course_id}: {str(e)}")
        return []

def fetch_tab_content(course_id: str, tab: Dict[str, Any], date_range: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Fetch content for a single tab.
    
    Args:
        course_id: The Canvas course ID
        tab: Tab information dictionary
        date_range: Optional date range filter
        
    Returns:
        Tuple of (tab_key, content_dict)
    """
    tab_key = tab.get("label", "").lower()
    try:
        if tab_key == "people":
            users = fetch_course_users(course_id)
            content = {
                "users": users,
                "metadata": {
                    "total_users": len(users),
                    "role_counts": {}
                }
            }
            # Count users by role
            for user in users:
                role = user.get("role", "Unknown")
                content["metadata"]["role_counts"][role] = content["metadata"]["role_counts"].get(role, 0) + 1
            return tab_key, content
            
        elif tab_key == "modules":
            try:
                modules_content = get_module_content(course_id)
                if modules_content:
                    content = {
                        "modules": modules_content,
                        "metadata": {
                            "total_modules": modules_content.count("## Module:") if isinstance(modules_content, str) else 0
                        }
                    }
                else:
                    content = {"modules": [], "metadata": {"total_modules": 0}}
            except Exception as e:
                logger.error(f"Error fetching modules for course {course_id}: {str(e)}")
                content = {"modules": [], "metadata": {"total_modules": 0}}
            return tab_key, content
            
        elif tab_key == "announcements":
            announcements = fetch_announcements(course_id)
            if announcements and date_range:
                announcements = filter_by_date_range(announcements, date_range)
            content = {
                "announcements": announcements,
                "metadata": {
                    "total_announcements": len(announcements) if announcements else 0
                }
            }
            return tab_key, content
            
        elif tab_key == "home":
            try:
                home_content = get_home_content(course_id)
                if home_content:
                    content = {
                        "home": home_content,
                        "metadata": {
                            "content_length": len(home_content),
                            "has_content": bool(home_content.strip())
                        }
                    }
                else:
                    # Fallback to course details
                    url = f"{BASE_URL}/courses/{course_id}"
                    course_details = fetch_api_data(url)
                    if course_details:
                        content_parts = []
                        if course_details.get("name"):
                            content_parts.append(f"Course: {course_details['name']}")
                        if course_details.get("course_code"):
                            content_parts.append(f"Code: {course_details['course_code']}")
                        if course_details.get("term"):
                            content_parts.append(f"Term: {course_details['term']['name']}")
                        if course_details.get("description"):
                            content_parts.append(f"\nDescription:\n{clean_html(course_details['description'])}")
                        
                        content = {
                            "home": "\n".join(content_parts),
                            "metadata": {
                                "content_type": "course_details",
                                "has_content": True
                            }
                        }
                    else:
                        content = {
                            "home": "No home page content available.",
                            "metadata": {
                                "content_type": "empty",
                                "has_content": False
                            }
                        }
            except Exception as e:
                logger.error(f"Error fetching home content for course {course_id}: {str(e)}")
                content = {
                    "home": "Error fetching home page content.",
                    "metadata": {
                        "error": str(e),
                        "has_content": False
                    }
                }
            return tab_key, content
            
        else:
            # For other tabs, try to get content from the API
            try:
                url = f"{BASE_URL}/courses/{course_id}/{tab_key}"
                content = fetch_api_data(url)
                return tab_key, {"content": content} if content else None
            except Exception as e:
                logger.error(f"Error fetching content for tab {tab_key} in course {course_id}: {str(e)}")
                return tab_key, None
                
    except Exception as e:
        logger.error(f"Error processing tab {tab_key} for course {course_id}: {str(e)}")
        return tab_key, {"error": str(e)}

def get_section_content(course_id: str, tab_key: str, date_range: Optional[str] = None) -> Dict[str, Any]:
    """
    Get content for a specific section/tab of a course using parallel processing.
    
    Args:
        course_id: The Canvas course ID
        tab_key: The tab key to fetch content for (e.g., 'home', 'syllabus', 'people')
        date_range: Optional date range filter for time-based content
        
    Returns:
        Dictionary containing the section content and metadata
    """
    try:
        if not course_id:
            return {"error": "Course ID is required"}
            
        if not tab_key:
            return {"error": "Tab key is required"}
            
        # Normalize tab key to lowercase for comparison
        tab_key_lower = tab_key.lower()
        
        # Get available tabs and validate the requested tab
        available_tabs = get_available_tabs(course_id)
        matching_tab = next((tab for tab in available_tabs if tab.get("label", "").lower() == tab_key_lower), None)
        
        if not matching_tab:
            return {"error": f"Tab '{tab_key}' not found in course {course_id}"}
        
        # Initialize response
        response = {
            "type": tab_key,
            "course_id": course_id,
            "content": None,
            "metadata": {}
        }
        
        # Fetch the content for the requested tab
        tab_key, content = fetch_tab_content(course_id, matching_tab, date_range)
        if content:
            response["content"] = content
            
        return response
        
    except Exception as e:
        logger.error(f"Error in get_section_content for tab {tab_key} in course {course_id}: {str(e)}")
        return {"error": f"Failed to fetch content: {str(e)}"}

def get_all_section_content(course_id: str, date_range: Optional[str] = None, max_workers: int = 4) -> Dict[str, Any]:
    """
    Get content for all sections/tabs of a course in parallel.
    
    Args:
        course_id: The Canvas course ID
        date_range: Optional date range filter for time-based content
        max_workers: Maximum number of concurrent workers
        
    Returns:
        Dictionary containing all sections' content and metadata
    """
    try:
        # Get available tabs
        available_tabs = get_available_tabs(course_id)
        if not available_tabs:
            return {"error": "No tabs found for course"}
            
        # Initialize response
        response = {
            "course_id": course_id,
            "sections": {},
            "metadata": {
                "total_tabs": len(available_tabs),
                "successful_fetches": 0,
                "failed_fetches": 0
            }
        }
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_tab = {
                executor.submit(fetch_tab_content, course_id, tab, date_range): tab
                for tab in available_tabs
            }
            
            # Process completed tasks
            for future in as_completed(future_to_tab):
                tab = future_to_tab[future]
                try:
                    tab_key, content = future.result()
                    if content and "error" not in content:
                        response["sections"][tab_key] = content
                        response["metadata"]["successful_fetches"] += 1
                    else:
                        response["metadata"]["failed_fetches"] += 1
                except Exception as e:
                    logger.error(f"Error processing tab {tab.get('label')}: {str(e)}")
                    response["metadata"]["failed_fetches"] += 1
        
        return response
        
    except Exception as e:
        logger.error(f"Error in get_all_section_content for course {course_id}: {str(e)}")
        return {"error": f"Failed to fetch content: {str(e)}"}

def fetch_assignments(course_id: str) -> str:
    """Fetch and format assignments for a course"""
    try:
        url = f"{BASE_URL}/courses/{course_id}/assignments"
        params = {
            "per_page": 50,
            "order_by": "due_at",
            "order": "asc",
            "include": ["submission", "description"]
        }
        
        try:
            assignments = get_cached_content(url, params)
            logger.info(f"Raw assignments data: {assignments}")  # Debug log
        except ContentError as e:
            if "rate_limit" in str(e).lower():
                logger.warning("Rate limit hit, waiting before retry...")
                time.sleep(1)  # Wait 1 second before retry
                assignments = get_cached_content(url, params)
            else:
                raise
                
        if not assignments:
            return "No assignments found for this course."
            
        formatted = []
        assignments_by_date = {}  # Group assignments by due date
        current_time = datetime.now(pytz.UTC)
        
        for item in assignments:
            try:
                name = item.get("name", "").strip()
                if not name:
                    continue
                    
                # Clean and validate name
                name = re.sub(r'[<>]', '', name)  # Remove potential HTML tags
                name = name.replace('&', '&amp;')  # Escape ampersands
                
                # Get basic assignment info
                description = clean_html(item.get("description", ""))
                
                # Handle due date
                due_str = "No due date"
                due_at = item.get("due_at")
                logger.info(f"Raw due_at for {name}: {due_at}")  # Debug log
                
                if due_at:
                    try:
                        parsed_date = parse_canvas_date(due_at)
                        if parsed_date:
                            # Format date with suffix
                            day = parsed_date.day
                            suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
                            due_str = parsed_date.strftime(f"%d{suffix} %B, %Y at %H:%M:%S UTC")
                            logger.info(f"Formatted date for {name}: {due_str}")  # Debug log
                            
                            # Group by date
                            date_key = parsed_date.date()
                            if date_key not in assignments_by_date:
                                assignments_by_date[date_key] = []
                            assignments_by_date[date_key].append({
                                "name": name,
                                "date": parsed_date,
                                "is_past": parsed_date < current_time
                            })
                    except (ValueError, AttributeError) as e:
                        logger.warning(f"Invalid date format for assignment {name}: {e}")
                
                points = item.get("points_possible", "N/A")
                if points == "N/A":
                    points_str = "Points not specified"
                else:
                    points_str = f"{points} points"
                
                # Get submission info if available
                submission = item.get("submission", {})
                if not isinstance(submission, dict):
                    submission = {}
                
                submitted = "Yes" if submission and submission.get("submitted_at") else "No"
                grade = submission.get("grade", "Not graded") if submission else "Not graded"
                
                # Format assignment details
                assignment_text = [
                    f"Assignment: {name}",
                    f"Due: {due_str}",
                    f"Points: {points_str}"
                ]
                
                if description:
                    # Truncate long descriptions
                    if len(description) > 500:
                        description = description[:497] + "..."
                    assignment_text.append(f"Description: {description}")
                
                if submitted == "Yes":
                    assignment_text.append(f"Status: Submitted ({grade})")
                else:
                    assignment_text.append("Status: Not submitted")
                
                formatted.append("\n".join(assignment_text))
                
            except Exception as e:
                logger.warning(f"Failed to process assignment {name}: {e}")
                continue
        
        if not formatted:
            return "No valid assignments found for this course."
            
        # Add summary of assignments by date
        summary = ["Assignment Summary:"]
        for date in sorted(assignments_by_date.keys()):
            assignments = assignments_by_date[date]
            # Format date with suffix
            day = date.day
            suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
            date_str = date.strftime(f"%d{suffix} %B, %Y")
            summary.append(f"\n{date_str}:")
            for assignment in sorted(assignments, key=lambda x: x["date"]):
                status = " (Past due)" if assignment["is_past"] else ""
                summary.append(f"- {assignment['name']}{status}")
        
        return "\n\n---\n\n".join(["\n".join(summary)] + formatted)
        
    except Exception as e:
        logger.error(f"Failed to fetch assignments: {e}")
        if "rate_limit" in str(e).lower():
            return "Rate limit exceeded. Please try again in a few seconds."
        return f"Error fetching assignments: {str(e)}"

def get_available_tabs(course_id: str) -> List[Dict[str, Any]]:
    """
    Get list of available tabs for a course.
    
    Args:
        course_id: The Canvas course ID
        
    Returns:
        List of tab dictionaries with id, type and html_url
    """
    try:
        url = f"{BASE_URL}/courses/{course_id}/tabs"
        tabs = fetch_api_data(url)
        
        if not isinstance(tabs, list):
            logger.error(f"Invalid response format for tabs in course {course_id}")
            return []
            
        # Filter out hidden tabs and ensure required fields
        visible_tabs = []
        for tab in tabs:
            if isinstance(tab, dict) and not tab.get("hidden", False):
                tab_info = {
                    "id": tab.get("id", ""),
                    "type": tab.get("type", ""),
                    "label": tab.get("label", ""),
                    "html_url": tab.get("html_url", "")
                }
                if tab_info["id"] and tab_info["label"]:
                    visible_tabs.append(tab_info)
        
        logger.info(f"Found {len(visible_tabs)} visible tabs in course {course_id}")
        return visible_tabs
        
    except Exception as e:
        logger.error(f"Error fetching tabs for course {course_id}: {str(e)}")
        return []

def filter_by_date_range(items: List[Dict[str, Any]], date_range: str) -> List[Dict[str, Any]]:
    """
    Filter items by date range.
    
    Args:
        items: List of items with dates
        date_range: Date range string (e.g. 'this_week', 'next_week', etc.)
        
    Returns:
        Filtered list of items
    """
    try:
        now = datetime.now(pytz.UTC)
        
        if date_range == "this_week":
            start = now - timedelta(days=now.weekday())
            end = start + timedelta(days=6)
        elif date_range == "next_week":
            start = now + timedelta(days=(7-now.weekday()))
            end = start + timedelta(days=6)
        elif date_range == "this_month":
            start = now.replace(day=1)
            end = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        else:
            return items
            
        filtered = []
        for item in items:
            if not isinstance(item, dict):
                continue
                
            date_str = item.get("due_at") or item.get("posted_at")
            if not date_str:
                continue
                
            try:
                date = parse_canvas_date(date_str)
                if start <= date <= end:
                    filtered.append(item)
            except ValueError:
                continue
                
        return filtered
    except Exception as e:
        logger.error(f"Error filtering by date range: {str(e)}")
        return items

def get_home_content(course_id: str) -> str:
    """
    Get course home page content.
    
    Args:
        course_id: The Canvas course ID
        
    Returns:
        Home page content as HTML string
    """
    try:
        # Try front page first
        url = f"{BASE_URL}/courses/{course_id}/front_page"
        content = fetch_api_data(url)
        if isinstance(content, dict) and content.get("body"):
            return clean_html(content["body"])
            
        # Fall back to course details
        url = f"{BASE_URL}/courses/{course_id}"
        content = fetch_api_data(url)
        if isinstance(content, dict):
            parts = []
            if content.get("name"):
                parts.append(f"<h1>{content['name']}</h1>")
            if content.get("course_code"):
                parts.append(f"<h2>{content['course_code']}</h2>")
            if content.get("description"):
                parts.append(clean_html(content["description"]))
            return "\n".join(parts)
            
        return ""
    except Exception as e:
        logger.error(f"Error fetching home content for course {course_id}: {str(e)}")
        return ""

def fetch_page_content(url: str) -> str:
    """
    Fetch content from a Canvas page URL.
    
    Args:
        url: The Canvas page URL
        
    Returns:
        Page content as HTML string
    """
    try:
        response = requests.get(url, headers=get_headers())
        if response.status_code == 200:
            return clean_html(response.text)
        return ""
    except Exception as e:
        logger.error(f"Error fetching page content from {url}: {str(e)}")
        return ""
