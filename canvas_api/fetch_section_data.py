import requests
from bs4 import BeautifulSoup
from canvas_api.auth import get_headers

BASE_URL = "https://canvas.illinois.edu/api/v1"

def get_section_data(course_id):
    url = f"{BASE_URL}/courses/{course_id}/tabs"
    response = requests.get(url, headers=get_headers())
    tabs = response.json()
    return {tab["label"]: {"id": tab["id"], "type": tab["type"], "position": tab["position"]} for tab in tabs}

def get_section_content(course_id, tab_key):
    headers = get_headers()

    def clean_html(html):
        soup = BeautifulSoup(html, "html.parser")
        if soup.iframe:
            return ""  # Skip iframe-only pages
        return soup.get_text(separator="\n").strip()

    try:
        # Debugging: show the tab being processed
        print(f"[DEBUG] Fetching content for tab: {tab_key}")

        if tab_key.lower() == "home":
            html = requests.get(f"https://canvas.illinois.edu/courses/{course_id}", headers=headers).text
            text = clean_html(html)
            if not text:
                print(f"[DEBUG] Skipping {tab_key} (empty or iframe)")
            return text

        elif tab_key.lower() == "syllabus":
            html = requests.get(f"https://canvas.illinois.edu/courses/{course_id}/assignments/syllabus", headers=headers).text
            text = clean_html(html)
            if not text:
                print(f"[DEBUG] Skipping {tab_key} (empty or iframe)")
            return text

        elif tab_key.lower() == "announcements":
            url = f"{BASE_URL}/courses/{course_id}/announcements?per_page=3"
            response = requests.get(url, headers=headers)
            items = response.json()
            return "\n\n".join(item["title"] + "\n" + BeautifulSoup(item["message"], "html.parser").get_text() for item in items)

        elif tab_key.lower() == "assignments":
            url = f"{BASE_URL}/courses/{course_id}/assignments?per_page=5"
            response = requests.get(url, headers=headers)
            items = response.json()
            return "\n\n".join(item["name"] + "\n" + BeautifulSoup(item.get("description") or "", "html.parser").get_text() for item in items)

        elif tab_key.lower() == "modules":
            url = f"{BASE_URL}/courses/{course_id}/modules?per_page=3"
            response = requests.get(url, headers=headers)
            modules = response.json()

            module_texts = []
            for mod in modules:
                items_url = f"{BASE_URL}/courses/{course_id}/modules/{mod['id']}/items"
                items = requests.get(items_url, headers=headers).json()

                for item in items:
                    if item["type"] == "Page":
                        page_url = f"{BASE_URL}/courses/{course_id}/pages/{item['page_url']}"
                        page = requests.get(page_url, headers=headers).json()
                        body = BeautifulSoup(page.get("body", ""), "html.parser").get_text()
                        module_texts.append(item["title"] + "\n" + body)

            return "\n\n".join(module_texts)

        else:
            print(f"[DEBUG] Skipping {tab_key} (unsupported tab type)")
            return ""

    except Exception as e:
        print(f"[ERROR] Failed to fetch section {tab_key} for course {course_id}: {e}")
        return ""
