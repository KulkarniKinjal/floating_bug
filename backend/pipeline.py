"""
backend/pipeline.py
--------------------
Glues together every teammate's piece into one function the Streamlit app
calls. This is the ONLY new file needed to make app.py actually rank
candidates instead of just showing an upload confirmation.

Flow:
    uploaded files
        -> parse.parse_pdf()            (raw text)
        -> jd.structure_jd()             (JD text -> {must_have, nice_to_have, role_summary})
        -> keyword_matching.run_keyword_leg()   (BM25 keyword_score + skill_coverage)
        -> semantic.semantic_score()             (per-resume semantic_score)
        -> fusion.fuse_and_rank()                 (final_score, sorted best-first)

Nothing in here decides a score itself — it only calls each leg's own
function and merges the dicts by candidate_id.
"""

import os
from io import BytesIO

from backend.parse import parse_pdf
from backend.jd import structure_jd
from backend.keyword_matching import run_keyword_leg
from backend.semantic import semantic_score
from backend.fusion import fuse_and_rank


def _derive_candidate_id(uploaded_file, index):
    """
    Use the uploaded filename (minus extension) as the candidate_id, so the
    ranked table shows something recognizable instead of resume_01/02/03.
    Falls back to resume_{index} if the file has no usable name.
    """
    name = getattr(uploaded_file, "name", None)
    if name:
        return os.path.splitext(name)[0]
    return f"resume_{index:02d}"


def build_resumes_dict(resume_files):
    """
    Turn Streamlit's list of UploadedFile objects into the agreed shared
    format: {candidate_id: full_resume_text}.
    """
    resumes = {}
    for i, f in enumerate(resume_files, start=1):
        f.seek(0)  # UploadedFile's read position may have been consumed already
        text = parse_pdf(f)
        candidate_id = _derive_candidate_id(f, i)
        resumes[candidate_id] = text
    return resumes


def build_jd(jd_file):
    """
    Parse the uploaded JD file and run it through the LLM structuring step.
    Requires Ollama running locally with the model pulled
    (ollama pull llama3.2:1b) — this call will fail if Ollama isn't up.
    """
    jd_file.seek(0)
    jd_text = parse_pdf(jd_file)
    return structure_jd(jd_text)


def run_pipeline(jd_file, resume_files):
    """
    Full end-to-end run. Returns a list of candidate dicts, sorted best-first,
    each with: candidate_id, final_score, keyword_score, semantic_score,
    skill_coverage — exactly the agreed per-candidate contract.
    """
    resumes = build_resumes_dict(resume_files)
    jd = build_jd(jd_file)

    # Keyword leg: BM25 score + alias-aware skill_coverage with evidence
    keyword_results = run_keyword_leg(resumes, jd)

    # Semantic leg: one call per resume against the structured JD
    candidates = []
    for kw_result in keyword_results:
        candidate_id = kw_result["candidate_id"]
        resume_text = resumes[candidate_id]
        candidates.append({
            "candidate_id": candidate_id,
            "keyword_score": kw_result["keyword_score"],
            "semantic_score": semantic_score(resume_text, jd),
            "skill_coverage": kw_result["skill_coverage"],
        })

    # Fusion: adds final_score, returns sorted best-first
    ranked = fuse_and_rank(candidates)

    return ranked, jd