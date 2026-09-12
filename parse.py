# parse.py
import pdfplumber
import os

def parse_pdf(path):
    """Extract all text from one PDF as a single string."""
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def load_resumes(folder="resumes"):
    """Return the agreed format: {resume_id: full_text}."""
    resumes = {}
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(".pdf"):
            resume_id = os.path.splitext(filename)[0]   # "resume_01.pdf" -> "resume_01"
            resumes[resume_id] = parse_pdf(os.path.join(folder, filename))
    return resumes

def load_jd_text(path="jd.pdf"):
    """Raw JD text — jd.py will structure it."""
    return parse_pdf(path)

if __name__ == "__main__":
    resumes = load_resumes()
    print(f"Loaded {len(resumes)} resumes")
    for rid, text in resumes.items():
        print(f"  {rid}: {len(text)} chars")