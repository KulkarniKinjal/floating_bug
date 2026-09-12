import streamlit as st

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
        with st.spinner("Structuring JD and scoring resumes..."):
            ranked, jd = run_pipeline(jd_file, resume_files)

        st.success(f"Ranked {len(ranked)} candidates against the JD.")

        # Show what the JD was parsed into
        with st.expander("Job Description skills (parsed)"):
            st.write("**Must have:**", ", ".join(jd.get("must_have", [])))
            st.write("**Nice to have:**", ", ".join(jd.get("nice_to_have", [])))
            st.write("**Role:**", jd.get("role_summary", ""))

        # Full ranked table
        st.header("Ranked Candidates")
        table = [
            {
                "Rank": i,
                "Candidate": c["candidate_id"],
                "Final": c["final_score"],
                "Semantic": round(c["semantic_score"], 3),
                "Keyword": round(c["keyword_score"], 3),
            }
            for i, c in enumerate(ranked, 1)
        ]
        st.dataframe(table)

        # Top-3 explanations
        st.header("Top 3 — Why they ranked here")
        for i, c in enumerate(get_top_candidates(ranked, 3), 1):
            summary = get_skill_summary(c)
            matched = [m["skill"] for m in summary["matched"]]
            missing = summary["missing"]

            with st.expander(f"#{i}  {c['candidate_id']}  (score {c['final_score']})"):
                st.write("**Matched skills:**", ", ".join(matched) if matched else "None")
                st.write("**Missing skills:**", ", ".join(missing) if missing else "None")
                st.write("**Evidence:**")
                for m in summary["matched"]:
                    st.write(f"- *{m['skill']}*: {m['evidence']}")
