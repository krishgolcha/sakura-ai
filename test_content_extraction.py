import logging
from canvas_api.fetch_section_data import get_section_data, get_section_content
from canvas_api.auth import get_headers

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_course_sections(course_id: str):
    """Test content extraction for all sections of a course."""
    logger.info(f"\nTesting course: {course_id}")
    
    # Get available sections
    sections = get_section_data(course_id)
    if not sections:
        logger.error(f"No sections found for course {course_id}")
        return
    
    logger.info(f"Available sections: {list(sections.keys())}")
    
    # Test each section
    for section_name, section_info in sections.items():
        logger.info(f"\nTesting section: {section_name}")
        
        # Get section content
        content = get_section_content(course_id, section_name)
        
        # Analyze content
        if not content:
            logger.warning(f"No content found for {section_name}")
            continue
            
        # Print content summary
        lines = content.split('\n')
        logger.info(f"Content length: {len(content)} characters")
        logger.info(f"Number of lines: {len(lines)}")
        logger.info(f"First few lines:\n{content[:500]}...")
        
        # Check for common issues
        if len(content) < 50:
            logger.warning(f"Content seems too short for {section_name}")
        if 'iframe' in content.lower():
            logger.warning(f"Found iframe in {section_name} content")
        if 'script' in content.lower():
            logger.warning(f"Found script tag in {section_name} content")

if __name__ == "__main__":
    # Test with some example courses
    test_courses = [
        "12345",  # Replace with actual course ID
        "67890",  # Replace with actual course ID
    ]
    
    for course_id in test_courses:
        test_course_sections(course_id) 