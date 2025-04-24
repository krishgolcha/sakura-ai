import argparse
import sys
import logging
from typing import List, Dict
from canvas_api.fetch_course_data import get_all_courses
from canvas_api.fetch_section_data import get_section_data, get_section_content
from embedder import embed_text
from processing.retriever import save_faiss_index
from utils.text_splitter import split_text
from utils.logger import log_event

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def index_section(course_id: str, section: str, content: str) -> bool:
    """Index a single section's content."""
    try:
        if not content or len(content.strip()) < 50:
            logger.warning(f"Skipping {section} - insufficient content")
            return False

        # Split content into chunks
        chunks = split_text(content)
        if not chunks:
            logger.warning(f"No chunks generated for {section}")
            return False

        # Generate embeddings
        embeddings = embed_text(chunks)
        if not embeddings:
            logger.warning(f"No embeddings generated for {section}")
            return False

        # Save index
        save_faiss_index(embeddings, chunks, section, course_id)
        logger.info(f"✓ Indexed {section} successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to index {section}: {str(e)}")
        return False

def index_course(course_id: str, sections: List[str] = None) -> Dict[str, bool]:
    """Index all or specified sections of a course."""
    results = {}
    
    # Get available sections
    section_data = get_section_data(course_id)
    if not section_data:
        logger.error("No sections found for course")
        return results

    # Filter sections if specified
    available_sections = list(section_data.keys())
    if sections:
        available_sections = [s for s in sections if s in available_sections]

    # Index each section
    for section in available_sections:
        logger.info(f"\nIndexing {section}...")
        content = get_section_content(course_id, section)
        results[section] = index_section(course_id, section, content)

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Index Canvas course content for quick retrieval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --course-id 52692
  %(prog)s --course-id 52692 --sections Syllabus Assignments
  %(prog)s --list-courses
  %(prog)s --course-name "IS 327"
"""
    )

    parser.add_argument(
        "--course-id",
        type=str,
        help="Canvas course ID to index"
    )

    parser.add_argument(
        "--course-name",
        type=str,
        help="Course name or code to index (e.g., 'IS 327')"
    )

    parser.add_argument(
        "--sections",
        nargs="+",
        help="Specific sections to index (e.g., Syllabus Assignments)"
    )

    parser.add_argument(
        "--list-courses",
        action="store_true",
        help="List available courses"
    )

    args = parser.parse_args()

    # List courses if requested
    if args.list_courses:
        courses = get_all_courses()
        print("\nAvailable courses:")
        print("-" * 80)
        for course in courses:
            print(f"• [{course['id']}] {course.get('name', 'Unnamed Course')}")
        print("-" * 80)
        return

    # Get course ID
    course_id = args.course_id
    if args.course_name and not course_id:
        courses = get_all_courses()
        for course in courses:
            name = course.get('name', '').lower()
            if args.course_name.lower() in name:
                course_id = str(course['id'])
                break

    if not course_id:
        parser.error("Please provide either --course-id or --course-name")

    # Index the course
    results = index_course(course_id, args.sections)

    # Print summary
    print("\nIndexing Summary:")
    print("-" * 80)
    for section, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {section}")
    print("-" * 80)

if __name__ == "__main__":
    main() 