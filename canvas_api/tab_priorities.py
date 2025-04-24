"""
Tab priority information for Canvas course content analysis.
This information helps prioritize which tabs to search first based on the type of information being sought.
"""

TAB_PRIORITIES = {
    "Home": {
        "description": "Course overview and quick links",
        "content_types": {
            "overview": "High",
            "instructor_contact": "Medium",
            "quick_links": "Medium"
        }
    },
    "Modules": {
        "description": "Course structure and resources",
        "content_types": {
            "slides": "High",
            "resources": "High",
            "assignment_links": "Medium",
            "syllabus": "Low",
            "weekly_structure": "High"
        }
    },
    "Assignments": {
        "description": "Graded work and submissions",
        "content_types": {
            "graded_assignments": "High",
            "due_dates": "High",
            "rubrics": "Medium"
        }
    },
    "Syllabus": {
        "description": "Course policies and schedule",
        "content_types": {
            "syllabus_text": "High",
            "calendar": "Medium",
            "policies": "Medium"
        }
    },
    "Announcements": {
        "description": "Course updates and messages",
        "content_types": {
            "instructor_messages": "High",
            "deadline_reminders": "Medium",
            "class_changes": "Low"
        }
    },
    "Pages": {
        "description": "Custom course content",
        "content_types": {
            "lecture_notes": "Medium",
            "resource_links": "Medium",
            "policies": "Low"
        }
    },
    "Files": {
        "description": "Course materials and resources",
        "content_types": {
            "pdfs_slides": "High",
            "syllabus": "Medium",
            "resources": "Medium"
        }
    },
    "Discussions": {
        "description": "Course discussions and interactions",
        "content_types": {
            "student_threads": "High",
            "instructor_topics": "Medium",
            "peer_replies": "High"
        }
    },
    "Grades": {
        "description": "Student performance tracking",
        "content_types": {
            "gradebook": "High",
            "feedback": "Medium",
            "late_missing": "Medium"
        }
    },
    "Quizzes": {
        "description": "Assessments and tests",
        "content_types": {
            "timed_quizzes": "High",
            "practice_tests": "Medium",
            "module_links": "Medium"
        }
    }
}

def get_tab_priority_prompt(available_tabs: list) -> str:
    """
    Generate a concise prompt for Gemini about tab priorities based on available tabs.
    
    Args:
        available_tabs: List of available tab names for the course
        
    Returns:
        str: A formatted prompt string with tab priority information
    """
    prompt_parts = ["Available tabs and their likely content (priority in brackets):"]
    
    for tab in available_tabs:
        if tab in TAB_PRIORITIES:
            tab_info = TAB_PRIORITIES[tab]
            content_types = []
            for content_type, priority in tab_info["content_types"].items():
                content_types.append(f"{content_type} [{priority}]")
            
            prompt_parts.append(f"\n{tab}: {', '.join(content_types)}")
    
    prompt_parts.append("\nSearch tabs in order of priority. If data not found, move to next tab.")
    return "\n".join(prompt_parts) 