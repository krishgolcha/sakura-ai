import requests
import os
from canvas_api.auth import get_token

def get_syllabus_text(course_id: str) -> str:
    url = f"https://canvas.illinois.edu/api/v1/courses/{course_id}?include[]=syllabus_body"
    headers = {"Authorization": f"Bearer {get_token()}"}

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    return data.get("syllabus_body", "")
