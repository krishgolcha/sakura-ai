def select_best_file(question: str, files: list, section_data: dict):
    # Simple heuristic: keyword match in filename
    for f in files:
        if any(keyword in f["name"].lower() for keyword in ["syllabus", "schedule", "overview"]):
            return f

    # Fallback to first file
    return files[0] if files else {"path": "dummy.pdf", "name": "placeholder"}
