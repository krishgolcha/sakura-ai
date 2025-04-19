# canvas_api/fetch_course_data.py

import os
import requests
from canvas_api.auth import get_headers

BASE_URL = "https://canvas.illinois.edu/api/v1"

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
        url = f"{BASE_URL}/courses/{course_id}/syllabus"
        r = requests.get(url, headers=headers)
        return r.json().get("syllabus_body", "")
    
    elif tab_id == "modules":
        url = f"{BASE_URL}/courses/{course_id}/modules?include=items&per_page=100"
        r = requests.get(url, headers=headers)
        return "\n\n".join(item["title"] for mod in r.json() for item in mod.get("items", []))
    
    elif tab_id == "assignments":
        url = f"{BASE_URL}/courses/{course_id}/assignments?per_page=100"
        r = requests.get(url, headers=headers)
        return "\n\n".join(a.get("description") or "" for a in r.json())

    elif tab_id == "announcements":
        url = f"{BASE_URL}/announcements?context_codes[]=course_{course_id}&per_page=10"
        r = requests.get(url, headers=headers)
        return "\n\n".join(a.get("message") or "" for a in r.json())

    elif tab_id == "grades":
        return ""  # Usually non-textual

    elif tab_id == "people":
        return ""  # Also not relevant for QA

    return ""  # Unknown or unsupported tab
