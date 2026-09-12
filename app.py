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
        with st.spinner("Structuring JD and scoring resumes..."):
            ranked, jd = run_pipeline(jd_file, resume_files)

        # Save results so the comparison section below can use them
        st.session_state["ranked"] = ranked
        st.session_state["jd"] = jd

        st.success(f"Ranked {len(ranked)} candidates against the JD.")

# Show results if we have them (persists across reruns)
if "ranked" in st.session_state:
    ranked = st.session_state["ranked"]
    jd = st.session_state["jd"]

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
    st.dataframe(table, use_container_width=True)

    # Top-3 explanations
    st.header("Top 3 - Why they ranked here")
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

    # ---- Bonus: recruiter comparison chat ----
    st.header("Ask: Why is X ranked above Y?")
    names = [c["candidate_id"] for c in ranked]

    col1, col2 = st.columns(2)
    with col1:
        cand_x = st.selectbox("Candidate X", names, index=0)
    with col2:
        cand_y = st.selectbox("Candidate Y", names, index=min(1, len(names) - 1))

    if st.button("Explain the ranking"):
        cx = next(c for c in ranked if c["candidate_id"] == cand_x)
        cy = next(c for c in ranked if c["candidate_id"] == cand_y)

        def summarize(c):
            matched = [s for s, d in c["skill_coverage"].items() if d["present"]]
            missing = [s for s, d in c["skill_coverage"].items() if not d["present"]]
            return (f"{c['candidate_id']}: final={c['final_score']}, "
                    f"semantic={round(c['semantic_score'], 3)}, "
                    f"keyword={round(c['keyword_score'], 3)}. "
                    f"Matched skills: {matched}. Missing skills: {missing}.")

        facts = summarize(cx) + "\n" + summarize(cy)
        prompt = (
            "You are a recruiting assistant. Using ONLY the data below, explain in "
            "3-4 sentences why the first candidate is ranked relative to the second. "
            "Refer to their scores and specific matched/missing skills. "
            "Do not invent anything.\n\n" + facts
        )

        with st.spinner("Thinking..."):
            resp = ollama.chat(
                model="llama3.2:1b",
                messages=[{"role": "user", "content": prompt}],
            )
        st.write(resp["message"]["content"])