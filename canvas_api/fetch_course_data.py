# canvas_api/fetch_course_data.py

import os
import requests
from canvas_api.auth import get_headers
from typing import List, Dict, Optional
from datetime import datetime
import pytz
from cachetools import TTLCache
import dateutil.parser
import logging

logger = logging.getLogger(__name__)

BASE_URL = "https://canvas.illinois.edu/api/v1"

# Initialize cache
_cache = TTLCache(maxsize=100, ttl=3600)  # 1 hour default TTL

def get_cached_content(endpoint: str, ttl: int = 3600) -> Dict:
    """Fetch content from Canvas API with caching."""
    cache_key = f"{endpoint}:{ttl}"
    if cache_key in _cache:
        logger.info(f"Cache hit for {endpoint}")
        return _cache[cache_key]
    
    logger.info(f"Cache miss for {endpoint}, fetching from API")
    headers = get_headers()
    url = f"{BASE_URL}/{endpoint}"
    logger.info(f"Making API request to: {url}")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    content = response.json()
    
    logger.info(f"Got response for {endpoint}: {len(content)} items")
    _cache[cache_key] = content
    return content

def parse_canvas_date(date_str: str) -> Optional[datetime]:
    """
    Parse Canvas date string to datetime object.
    
    Args:
        date_str: The date string from Canvas
        
    Returns:
        datetime: A timezone-aware datetime object, or None if the date is invalid/empty
    """
    if not date_str:
        return None
    try:
        dt = dateutil.parser.parse(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        return dt
    except (ValueError, TypeError):
        return None

def get_all_courses():
    """Fetch all enrolled Canvas courses for the user."""
    headers = get_headers()
    url = f"{BASE_URL}/courses?per_page=100"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def get_section_content(course_id: str, tab_id: str):
    """Fetch HTML content from a specific course tab."""
    headers = get_headers()

    if tab_id == "home":
        url = f"{BASE_URL}/courses/{course_id}/front_page"
        r = requests.get(url, headers=headers)
        return r.json().get("body", "")
    
    elif tab_id == "syllabus":
        try:
            # First try to get the syllabus body directly
            url = f"{BASE_URL}/courses/{course_id}?include[]=syllabus_body"
            r = requests.get(url, headers=headers)
            syllabus_body = r.json().get("syllabus_body")
            
            if syllabus_body:
                return clean_html(syllabus_body)
                
            # If that fails, try the syllabus endpoint
            url = f"{BASE_URL}/courses/{course_id}/syllabus"
            r = requests.get(url, headers=headers)
            data = r.json()
            
            if isinstance(data, dict):
                content_parts = []
                # Try different possible fields where syllabus content might be
                if data.get("syllabus_body"):
                    content_parts.append(clean_html(data["syllabus_body"]))
                if data.get("body"):
                    content_parts.append(clean_html(data["body"]))
                if data.get("description"):
                    content_parts.append(clean_html(data["description"]))
                    
                return "\n\n".join(part for part in content_parts if part)
            
            return ""
            
        except Exception as e:
            logger.error(f"Error fetching syllabus: {e}")
            return ""
    
    elif tab_id == "modules":
        url = f"{BASE_URL}/courses/{course_id}/modules?include=items&per_page=100"
        r = requests.get(url, headers=headers)
        return "\n\n".join(item["title"] for mod in r.json() for item in mod.get("items", []))
    
    elif tab_id == "assignments":
        url = f"{BASE_URL}/courses/{course_id}/assignments?per_page=100"
        r = requests.get(url, headers=headers)
        return "\n\n".join(clean_html(a.get("description") or "") for a in r.json())

    elif tab_id == "announcements":
        url = f"{BASE_URL}/announcements?context_codes[]=course_{course_id}&per_page=10"
        r = requests.get(url, headers=headers)
        return "\n\n".join(clean_html(a.get("message") or "") for a in r.json())

    elif tab_id == "grades":
        return ""  # Usually non-textual

    elif tab_id == "people":
        return ""  # Also not relevant for QA

    return ""  # Unknown or unsupported tab

def fetch_assignments(course_id: str) -> List[Dict]:
    """Fetch all assignments and graded discussions for a course."""
    logger.info(f"Fetching assignments for course {course_id}")
    try:
        assignments = get_cached_content(f'courses/{course_id}/assignments', ttl=1800)
        logger.info(f"Found {len(assignments)} assignments")
        logger.debug("Raw assignments:")
        for a in assignments:
            logger.debug(f"- {a.get('name')}: due={a.get('due_at')}, published={a.get('published')}, points={a.get('points_possible')}")
        
        discussions = get_cached_content(f'courses/{course_id}/discussion_topics', ttl=900)
        logger.info(f"Found {len(discussions)} discussions")
        logger.debug("Raw discussions:")
        for d in discussions:
            logger.debug(f"- {d.get('name')}: due={d.get('due_at')}, published={d.get('published')}, points={d.get('points_possible')}")
        
        formatted_assignments = []
        
        # Process all assignments, regardless of published status
        for assignment in assignments:
            formatted = format_assignment(assignment)
            if formatted:
                formatted_assignments.append(formatted)
                logger.debug(f"Added assignment: {formatted['name']}")
            else:
                logger.debug(f"Skipped assignment: {assignment.get('name')}")
        
        # Process all discussions that have an assignment
        for discussion in discussions:
            if discussion.get('assignment'):
                discussion['type'] = 'graded_discussion'
                formatted = format_assignment(discussion)
                if formatted:
                    formatted_assignments.append(formatted)
                    logger.debug(f"Added discussion: {formatted['name']}")
                else:
                    logger.debug(f"Skipped discussion: {discussion.get('name')}")
        
        logger.info(f"Total formatted assignments: {len(formatted_assignments)}")
        logger.debug("Final formatted assignments:")
        for a in formatted_assignments:
            logger.debug(f"- {a.get('name')}: due={a.get('due_at')}, type={a.get('type')}")
        
        # Sort assignments, putting None dates at the end
        def sort_key(x):
            due_date = parse_canvas_date(x.get('due_at'))
            return (due_date if due_date else datetime.max.replace(tzinfo=pytz.UTC))
        
        return sorted(formatted_assignments, key=sort_key)
        
    except Exception as e:
        logger.error(f"Error fetching assignments: {e}")
        return []

def format_assignment(assignment: Dict) -> Optional[Dict]:
    """
    Format an assignment or graded discussion for consistent processing.
    
    Args:
        assignment (Dict): Raw assignment data from Canvas API
        
    Returns:
        Optional[Dict]: Formatted assignment data or None if invalid/should be skipped
    """
    if not assignment or not isinstance(assignment, dict):
        logger.debug("Skipping invalid assignment")
        return None
        
    # Basic assignment data
    formatted = {
        'id': assignment.get('id'),
        'name': assignment.get('name', 'Untitled Assignment'),
        'description': assignment.get('description'),
        'due_at': assignment.get('due_at'),
        'points_possible': assignment.get('points_possible', 'Not specified'),
        'html_url': assignment.get('html_url'),
        'submission_types': assignment.get('submission_types', []),
        'type': assignment.get('type', 'assignment')
    }
    
    # Include if:
    # 1. It has a due date, or
    # 2. It's a graded discussion, or
    # 3. It's published and worth points
    if (assignment.get('due_at') or 
        assignment.get('type') == 'graded_discussion' or
        (assignment.get('published', False) and assignment.get('points_possible'))):
        logger.debug(f"Including assignment: {formatted['name']} (due: {formatted['due_at']}, type: {formatted['type']})")
        return formatted
        
    logger.debug(f"Skipping assignment: {formatted['name']} (no due date/not graded discussion/not published with points)")
    return None
