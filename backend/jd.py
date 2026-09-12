# jd.py
import ollama
import json

def structure_jd(jd_text, model="llama3.2:1b"):
    """Use a local LLM to turn raw JD text into structured skills.
    This ORGANIZES the input — it does not score anything."""
    prompt = f"""Extract skills from this job description. Return ONLY valid JSON, no other text.

Format:
{{
  "must_have": ["skill1", "skill2"],
  "nice_to_have": ["skill3"],
  "role_summary": "one sentence"
}}

Job description:
{jd_text}
"""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json"    # forces valid JSON output
    )
    content = response["message"]["content"]
    raw = json.loads(content)

    # Normalize keys — small models don't always spell them exactly right
    def find_key(d, *options):
        for k in d:
            normalized = k.lower().replace("-", "_").replace(" ", "_")
            if normalized in options:
                return d[k]
        return []

    return {
        "must_have": find_key(raw, "must_have", "musthave", "required", "required_skills"),
        "nice_to_have": find_key(raw, "nice_to_have", "nicetohave", "nice_to_haves", "preferred", "optional"),
        "role_summary": find_key(raw, "role_summary", "rolesummary", "summary") or ""
    }

if __name__ == "__main__":
    sample_jd = """Junior Full Stack Developer Intern at TechNova Solutions.
    We are looking for someone with experience in Node.js, MongoDB, and building
    REST APIs. Familiarity with React is required. Knowledge of Docker and AWS
    is a plus. Must know JavaScript and Git."""

    result = structure_jd(sample_jd)
    print(json.dumps(result, indent=2))