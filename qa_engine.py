from processing.course_matcher import match_course_name
from canvas_api.fetch_section_data import get_section_data
from canvas_api.fetch_files import get_files_for_course
from canvas_api.fetch_modules import get_module_content
from canvas_api.fetch_syllabus import get_syllabus_text
from processing.file_selector import select_best_file
from processing.file_parser import extract_text_from_file
from utils.prompt_loader import load_prompt
from utils.logger import log_event
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def answer_question(question: str, course_name: str = None) -> str:
    log_event("Received question", question)

    # Step 1: Match course
    matched_course = match_course_name(course_name or question)
    if not matched_course:
        raise ValueError(f"❌ No matching course found for input: {course_name or question}")
    course_id = matched_course["id"]
    log_event("Matched course", matched_course["name"])

    # Step 2: Try pulling from each section in order
    section_data = get_section_data(course_id)
    available_tabs = [tab["label"] for tab in section_data]
    log_event("Available sections", ", ".join(available_tabs))

    for section in ["Files", "Modules", "Syllabus"]:
        try:
            file_text = ""
            if section == "Files":
                files = get_files_for_course(course_id)
                if not files:
                    continue
                selected_file = select_best_file(question, files, section_data)
                log_event("Selected file", selected_file["name"])
                file_text = extract_text_from_file(selected_file["path"])
            elif section == "Modules":
                file_text = get_module_content(course_id)
            elif section == "Syllabus":
                file_text = get_syllabus_text(course_id)

            if file_text:
                break  # Stop once we've found usable content
        except Exception as e:
            print(f"[WARN] Failed on {section}: {e}")
            continue

    if not file_text:
        return "❌ Could not retrieve any usable content from Files, Modules, or Syllabus."

    # Step 3: Format GPT prompt
    prompt = load_prompt("answer_generation_prompt.txt", {
        "question": question,
        "file_text": file_text
    })

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering questions about a Canvas course."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ GPT call failed: {e}"