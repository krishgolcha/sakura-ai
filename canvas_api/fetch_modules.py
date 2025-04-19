import requests
import os
from canvas_api.auth import get_token

def get_module_content(course_id: str) -> str:
    headers = {"Authorization": f"Bearer {get_token()}"}

    # Step 1: Get all modules
    url = f"https://canvas.illinois.edu/api/v1/courses/{course_id}/modules?per_page=100"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    modules = response.json()

    full_text = ""

    # Step 2: For each module, get its items
    for mod in modules:
        mod_title = mod["name"]
        full_text += f"\n## Module: {mod_title}\n"

        mod_items_url = f"https://canvas.illinois.edu/api/v1/courses/{course_id}/modules/{mod['id']}/items"
        items_response = requests.get(mod_items_url, headers=headers)
        items_response.raise_for_status()
        items = items_response.json()

        for item in items:
            item_title = item["title"]
            full_text += f"- {item_title}\n"

    return full_text
