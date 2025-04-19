from utils.gpt import ask_gpt

def ask_gpt_section_ranker(question: str, section_data: dict) -> list:
    allowed = [tab for tab, meta in section_data.items() if meta.get("type") == "internal"]

    if not allowed:
        return []

    descriptions = {
        "Home": "Landing page (may include welcome message, course desc, instructor contact, meeting times, Zoom link, syllabus (sometimes), grading (sometimes), etc.)",
        "Syllabus": "Course overview, grading policy, and key dates",
        "Announcements": "Instructor messages, reminders, and schedule updates",
        "Assignments": "Homework, submission links, deadlines, and feedback",
        "Modules": "Week-by-week or topic-based learning units and content",
        "Files": "Lecture slides, PDFs, references, and downloadable materials",
        "Discussions": "Forums for student-instructor and student-student engagement",
        "Grades": "Grading dashboard showing scores and progress",
        "Pages": "Custom instructional pages made by the instructor",
        "People": "List of classmates, instructors, and TAs"
    }

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
