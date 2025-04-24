from qa_engine_rag import answer_question_rag
from canvas_api.fetch_course_data import get_all_courses
from processing.course_matcher import match_course_name_gpt
import time

def test_edge_cases():
    # Get all courses first
    all_courses = get_all_courses()
    if not all_courses:
        print("❌ No courses found. Please check your Canvas API credentials.")
        return

    # Test cases for each tab
    test_cases = {
        "Home": [
            "What is this course about?",
            "When are the office hours?",
            "What is the course schedule?",
            "How do I contact the instructor?",
            "Is there a Zoom link for the class?"
        ],
        "Syllabus": [
            "What is the grading policy?",
            "What are the course prerequisites?",
            "What textbooks do I need?",
            "What is the late submission policy?",
            "What are the course objectives?"
        ],
        "Announcements": [
            "What was the last announcement about?",
            "Has there been any schedule changes?",
            "Are there any upcoming deadlines?",
            "What was the last important update?",
            "Has the instructor posted any recent messages?"
        ],
        "Assignments": [
            "What is the next assignment?",
            "When is the next homework due?",
            "What are the submission requirements?",
            "How do I submit my work?",
            "What is the grading criteria for assignments?"
        ],
        "Modules": [
            "What topics are covered in the first module?",
            "What are the learning objectives for this week?",
            "What materials do I need to review?",
            "What are the key concepts in the current module?",
            "What activities are included in the modules?"
        ]
    }

    # Test each course
    for course in all_courses[:1]:  # Limit to first course for testing
        print(f"\n🔍 Testing course: {course['name']}")
        
        for tab, questions in test_cases.items():
            print(f"\n📑 Testing {tab} tab:")
            for question in questions:
                try:
                    print(f"\nQ: {question}")
                    answer = answer_question_rag(question)
                    print(f"A: {answer}")
                    time.sleep(2)  # Rate limiting
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
                    time.sleep(2)

if __name__ == "__main__":
    test_edge_cases()
