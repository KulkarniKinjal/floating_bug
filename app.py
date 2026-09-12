import streamlit as st
import ollama

from backend.pipeline import run_pipeline
from backend.ranking import get_top_candidates
from backend.explanation import get_skill_summary


st.title("Smart Shortlisting Engine")
st.write("Rank resumes against a Job Description")

st.header("Upload Files")

jd_file = st.file_uploader("Upload Job Description", type=["pdf"])
resume_files = st.file_uploader(
    "Upload Resumes", type=["pdf"], accept_multiple_files=True
)

if st.button("Rank Candidates"):
    if jd_file is None:
        st.warning("Please upload a Job Description.")
    elif not resume_files:
        st.warning("Please upload at least one resume.")
    else:
        st.success(
            f"JD uploaded + {len(resume_files)} resumes uploaded."
        )