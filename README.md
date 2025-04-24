# Sakura AI - Canvas Course Assistant

Sakura AI is an intelligent assistant that helps you get quick answers about your Canvas courses using natural language questions.

## Features

- Ask natural language questions about your courses
- Get information from course home pages, syllabi, announcements, assignments, and modules
- List all available courses
- Configurable through environment variables or config file
- Detailed error messages and debugging support

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sakura-ai.git
   cd sakura-ai
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your configuration either through:

   a. Environment variables:
   ```bash
   export CANVAS_API_TOKEN="your_canvas_token"
   export GOOGLE_API_KEY="your_gemini_api_key"
   ```

   b. Or create a config.json file:
   ```bash
   cp config.json.example config.json
   # Edit config.json with your tokens
   ```

5. Index your courses:
   ```bash
   # List available courses with their IDs
   python index_course.py --list-courses

   # Index a course by ID
   python index_course.py --course-id YOUR_COURSE_ID

   # Or index by course name
   python index_course.py --course-name "IS 327"

   # Index specific sections only
   python index_course.py --course-name "IS 327" --sections Syllabus Assignments
   ```

## Usage

1. List your available courses:
   ```bash
   python ask.py --list-courses
   ```

2. Ask questions about your courses:
   ```bash
   python ask.py "What are the office hours for CS 225?"
   python ask.py "When is the next assignment due in IS 327?"
   python ask.py "What is the late submission policy?"
   ```

3. Debug mode for more detailed output:
   ```bash
   python ask.py --debug "What is the grading policy?"
   ```

## Examples

```bash
# List all courses
$ python ask.py --list-courses

# Index a course
$ python index_course.py --course-name "IS 327"

# Ask about office hours
$ python ask.py "When are the office hours for CS 225?"

# Check assignment deadlines
$ python ask.py "What assignments are due this week in IS 327?"

# Look up course policies
$ python ask.py "What is the late submission policy for CS 225?"
```

## Supported Canvas Sections

The assistant can search through and answer questions about:
- Course Home pages
- Syllabus
- Announcements
- Assignments
- Modules

## Troubleshooting

1. If you get a "Missing required configuration" error:
   - Check that your Canvas API token and Google API key are properly set
   - Verify your config.json file or environment variables

2. If you get "No courses found":
   - Verify your Canvas API token has the correct permissions
   - Check your Canvas API URL is correct

3. If you get "Missing FAISS index" error:
   - Make sure you've indexed the course first using index_course.py
   - Try re-indexing the course if the content has changed

4. For other issues:
   - Run the command with --debug flag for more detailed error messages
   - Check the logs for specific error information

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your chosen license] 