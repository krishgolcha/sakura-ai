# Standard library imports
import logging
from typing import List, Optional
from functools import lru_cache

# Third-party imports
import google.generativeai as genai

# Local imports
from processing.course_matcher import match_course_name_gpt, get_all_courses, get_course_id
from canvas_api.fetch_section_data import get_section_data
from processing.retriever import retrieve_chunks
from processing.section_picker import ask_gpt_section_ranker
from utils.prompt_loader import load_prompt
from utils.logger import log_event
from utils.gpt import ask_gpt
from processing.gemini_client import init_gemini, get_gemini_response

# Initialize Gemini
init_gemini()

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_gemini_model():
    """Get cached instance of Gemini model"""
    return genai.GenerativeModel('gemini-1.5-pro')

def get_answer_gemini(query: str, context: List[str], course_name: Optional[str] = None) -> str:
    """Get answer using Gemini"""
    try:
        # Format context
        formatted_context = "\n\n".join([f"Context {i+1}:\n{c}" for i, c in enumerate(context)])
        
        # Construct prompt
        prompt = f"""
You are a helpful teaching assistant for {course_name if course_name else 'a university course'}.
Answer the following question based on the provided context.

Context:
{formatted_context}

Question:
{query}

Guidelines:
1. Answer based only on the provided context
2. If context is insufficient, say so clearly
3. Format dates consistently like 24th March, 2025 instead of 2025-03-24.
4. Keep responses concise but complete
5. For assignments, include due dates and points
6. For announcements, include post dates
7. Use bullet points for lists
8. Cite specific context numbers when possible
9. Present information confidently without uncertainty
10. Include specific details like room numbers, times, and dates
11. Always mention the source (tab name) where information was found

Answer:"""

        # Use cached model
        model = get_gemini_model()
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1000,
                top_p=0.8,
                top_k=40
            )
        )
        
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Error getting answer: {e}")
        return "I encountered an error while processing your question. Please try again."

@lru_cache(maxsize=100)
def extract_course_name(question: str) -> Optional[str]:
    """Extract course name from question using course matcher with caching"""
    try:
        all_courses = get_all_courses()
        matched_course = match_course_name_gpt(question, all_courses)
        if matched_course:
            return matched_course["name"]
        return None
    except Exception as e:
        logger.error(f"Error extracting course name: {e}")
        return None

@lru_cache(maxsize=100)
def get_section_data_cached(course_id: str):
    """Cached wrapper for get_section_data"""
    return get_section_data(course_id)

def get_section_from_question(question: str) -> Optional[str]:
    """Get relevant section (tab) from question using GPT"""
    try:
        # Get available sections for the course
        course_name = extract_course_name(question)
        if not course_name:
            return None
            
        course_id = get_course_id(course_name)
        if not course_id:
            return None
            
        section_data = get_section_data_cached(str(course_id))
        if not section_data:
            return None
            
        # Ask GPT to identify most relevant section
        ranked_tabs = ask_gpt_section_ranker(question, section_data)
        if ranked_tabs:
            return ranked_tabs[0]  # Return highest ranked section
        return None
        
    except Exception as e:
        logger.error(f"Error getting section from question: {e}")
        return None

def get_relevant_context(question: str, course_name: str, section: Optional[str] = None) -> str:
    """Get relevant context chunks for the question"""
    try:
        course_id = get_course_id(course_name)
        if not course_id:
            return ""
            
        if section:
            # Search in specific section
            chunks = retrieve_chunks(question, section, str(course_id))
            if chunks:
                return "\n\n".join(chunks)
        else:
            # Try all sections
            section_data = get_section_data(str(course_id))
            if not section_data:
                return ""
                
            for section in section_data.keys():
                chunks = retrieve_chunks(question, section, str(course_id))
                if chunks:
                    return "\n\n".join(chunks)
        return ""
        
    except Exception as e:
        logger.error(f"Error getting relevant context: {e}")
        return ""

def answer_question_rag(question: str) -> str:
    """Answer a question using RAG with Gemini"""
    try:
        # Get course name and section from question
        course_name = extract_course_name(question)
        section = get_section_from_question(question)
        
        if not course_name:
            return "Please specify a course name in your question."
            
        # Get relevant context chunks
        context = get_relevant_context(question, course_name, section)
        
        if not context:
            return "I couldn't find any relevant information to answer your question."
            
        # Get answer using Gemini
        answer = get_answer_gemini(question, context, course_name)
        
        return answer
        
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {e}")
        return "I encountered an error while processing your question. Please try again."
