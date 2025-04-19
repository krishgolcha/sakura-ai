from processing.course_matcher import match_course_name_gpt
from canvas_api.fetch_course_data import get_all_courses
from canvas_api.fetch_section_data import get_section_data
from retriever import retrieve_chunks
from utils.prompt_loader import load_prompt
from utils.logger import log_event
from utils.gpt import ask_gpt
from processing.section_picker import ask_gpt_section_ranker

def answer_question_rag(question: str) -> str:
    # Step 1: Match course name
    all_courses = get_all_courses()
    matched_course = match_course_name_gpt(question, all_courses)
    if not matched_course:
        raise ValueError("❌ Could not match a course from your input.")

    course_id = str(matched_course["id"])
    course_name = matched_course["name"]
    log_event("Matched course", course_name)

    # Step 2: Get all Canvas sections (tabs)
    section_data = get_section_data(course_id)
    if not section_data:
        raise ValueError("❌ Could not fetch Canvas tabs for the selected course.")

    available_tabs = list(section_data.keys())
    log_event("Available tabs", available_tabs)

    # Step 3: Ask GPT to rank best tabs to search
    ranked_tabs = ask_gpt_section_ranker(question, section_data)
    log_event("GPT-ranked tabs", ranked_tabs)

    # Step 4: Search each ranked tab until answer is found
    for section in ranked_tabs:
        try:
            log_event("Trying section", section)
            chunks = retrieve_chunks(question, section, course_id)
            if not chunks:
                continue

            file_text = "\n\n".join(chunks)
            prompt = load_prompt("answer_generation_prompt.txt", {
                "question": question,
                "file_text": file_text,
                "course_name": course_name
            })

            answer = ask_gpt(prompt)
            if answer and len(answer.strip()) > 20:
                log_event("Generated answer", answer)
                return answer

        except Exception as e:
            log_event("Section error", str(e))
            continue

    return "We searched all likely sections but couldn't find an answer. Try rephrasing your question or check the Canvas course directly."
