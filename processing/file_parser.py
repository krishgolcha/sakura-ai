import os

def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_pdf(file_path)
    elif ext == ".docx":
        return extract_docx(file_path)
    elif ext == ".html":
        return extract_html(file_path)
    else:
        return extract_txt(file_path)

def extract_pdf(path):
    from PyPDF2 import PdfReader
    reader = PdfReader(path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_docx(path):
    import docx
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_html(path):
    from bs4 import BeautifulSoup
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        return soup.get_text()

def extract_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
