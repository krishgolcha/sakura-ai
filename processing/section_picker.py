from utils.gpt import ask_gpt
from typing import List
import json
import logging
from utils.gemini import get_gemini_response
from canvas_api.fetch_section_data import get_available_tabs

def ask_gpt_section_ranker(question: str, section_data: dict) -> list:
    allowed = [tab for tab, meta in section_data.items() if meta.get("type") == "internal"]

    if not allowed:
        return []

    descriptions = {
        "Home": "Landing page (may include welcome message, course desc, instructor contact, meeting times, Zoom link, syllabus (sometimes), grading (sometimes), etc.)",
        "Syllabus": "Course overview, policies, office hours, instructor contact, grading policy, and key dates",
        "Announcements": "Instructor messages, reminders, and schedule updates",
        "Assignments": "Homework, submission links, deadlines, and feedback",
        "Modules": "Week-by-week or topic-based learning units and content",
        "Files": "Lecture slides, PDFs, references, and downloadable materials",
        "Discussions": "Forums for student-instructor and student-student engagement",
        "Grades": "Grading dashboard showing scores and progress",
        "Pages": "Custom instructional pages made by the instructor",
        "People": "List of classmates, instructors, and TAs"
    }

    # Add keywords for better matching
    keywords = {
        "Syllabus": ["office hours", "instructor", "professor", "teacher", "contact", "policy", "policies", "grading", "schedule", "attendance"],
        "Home": ["welcome", "overview", "introduction", "about", "zoom", "meeting"],
        "Assignments": ["homework", "assignment", "due", "deadline", "submit", "quiz", "exam"],
        "Modules": ["lecture", "week", "topic", "content", "material", "reading"],
        "Announcements": ["announcement", "update", "news", "reminder"],
        "Discussions": ["discussion", "forum", "post", "thread", "reply"],
        "Grades": ["grade", "score", "point", "mark", "evaluation"],
        "People": ["classmate", "student", "instructor", "ta", "teaching assistant"]
    }

    # Check if question contains keywords for any section
    question_lower = question.lower()
    matched_sections = []
    for section, words in keywords.items():
        if section in allowed and any(word in question_lower for word in words):
            matched_sections.append(section)

    # If we found matches based on keywords, return those
    if matched_sections:
        return matched_sections[:3]

    # Otherwise, use GPT to rank sections
    tab_descriptions = "\n".join(
        f"- {tab}: {descriptions.get(tab, 'No description available')}" for tab in allowed
    )

    prompt = f"""
A student asked: "{question}"

Below are the available Canvas sections (tabs) for their course, along with descriptions:

{tab_descriptions}

Which **1–3 sections** are most likely to contain the answer? Return them as a Python list, like:
["Home", "Assignments"]
    """

    response = ask_gpt(prompt)
    try:
        ranked = eval(response.strip())
        return [r for r in ranked if r in allowed]
    except:
        return allowed[:1]

def get_relevant_tabs(course_id: int, question: str) -> List[str]:
    """Get the top 3 most relevant tabs for a given question."""
    try:
        # Get available tabs for validation
        available_tabs = get_available_tabs(course_id)
        if not available_tabs:
            logging.warning("No available tabs found")
            return []

        # Construct prompt for Gemini
        prompt = f"""Given the question "{question}", return a JSON list of the 3 most relevant Canvas LMS tabs from these options: {available_tabs}
        Format the response as a JSON list of strings, e.g. ["Assignments", "Modules", "Home"]"""

        # Get response from Gemini
        response = get_gemini_response(prompt, temperature=0.1)
        if not response:
            logging.warning("No response from Gemini")
            return []

        try:
            # Parse JSON response
            ranked_tabs = json.loads(response)
            if not isinstance(ranked_tabs, list):
                raise ValueError("Response is not a list")
            
            # Validate and filter tabs
            valid_tabs = [tab for tab in ranked_tabs if tab in available_tabs]
            if not valid_tabs:
                logging.warning("No valid tabs found in response")
                return []

            # Take top 3 valid tabs
            result = valid_tabs[:3]
            logging.info(f"Ranked tabs: {result}")
            return result

        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON response: {response}")
            return []
        except ValueError as e:
            logging.error(f"Invalid response format: {e}")
            return []

    except Exception as e:
        logging.error(f"Error in get_relevant_tabs: {e}")
        return []
