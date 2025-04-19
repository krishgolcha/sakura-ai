import difflib
import json
from utils.gpt import ask_gpt
from canvas_api.fetch_course_data import get_all_courses

def match_course_name_gpt(query, courses=None):
    if courses is None:
        courses = get_all_courses()

    # Try fuzzy match first
    query_norm = query.lower().replace(" ", "")
    best_match = None
    highest_score = 0

    for course in courses:
        name = course.get("name", "")
        code = course.get("course_code", "")
        name_norm = name.lower().replace(" ", "")
        code_norm = code.lower().replace(" ", "")

        if query_norm in name_norm or query_norm in code_norm:
            return course  # Direct substring match

        score = difflib.SequenceMatcher(None, query_norm, name_norm).ratio()
        if score > highest_score:
            best_match = course
            highest_score = score

    if highest_score > 0.8:
        return best_match

    # Fuzzy match failed – fallback to GPT
    print("[DEBUG] No fuzzy match found. Trying GPT fallback...")

    course_list_str = "\n".join(f"{c['id']}: {c.get('name', 'Unnamed Course')}" for c in courses)

    prompt = f"""
A student entered: "{query}"

Here is a list of their Canvas courses:

{course_list_str}

Please return ONLY the ID (e.g., 52669) of the course they are referring to, based on the list above.
If you're unsure, return "None".
"""

    response = ask_gpt(prompt)
    course_id = response.strip()
    print(f"[DEBUG] GPT returned: '{course_id}'")

    if course_id.lower() == "none":
        return None

    for c in courses:
        if str(c["id"]) == course_id:
            return c

    print("[DEBUG] GPT match failed.")
    return None
