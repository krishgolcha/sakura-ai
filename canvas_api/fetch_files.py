import os
import json
import requests
from canvas_api.auth import get_token

def get_files_for_course(course_id: str):
    path = f"data/files_{course_id}.json"

    # ✅ Load from cache if it exists
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)

    # 🛰 Fetch live from Canvas
    url = f"https://canvas.illinois.edu/api/v1/courses/{course_id}/files"
    headers = {
        "Authorization": f"Bearer {get_token()}"
    }
    params = {
        "per_page": 100
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 403:
            print(f"[ERROR] Access denied to course {course_id} files. Skipping file access.")
            return []
        else:
            raise

    files = response.json()

    # 💾 Cache for later
    os.makedirs("data", exist_ok=True)
    with open(path, "w") as f:
        json.dump(files, f, indent=2)

    return files
