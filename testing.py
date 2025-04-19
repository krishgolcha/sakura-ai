import requests
import os

token = os.getenv("CANVAS_API_TOKEN")
course_id = "52669"
url = f"https://canvas.illinois.edu/api/v1/courses/{course_id}/files"

headers = {"Authorization": f"Bearer {token}"}
params = {"per_page": 100}

r = requests.get(url, headers=headers, params=params)
print(r.status_code)
print(r.text)
