# semantic.py
from functools import lru_cache

from sentence_transformers import SentenceTransformer, util


@lru_cache(maxsize=1)
def get_model():
    """Load the embedding model on first ranking request, then reuse it."""
    return SentenceTransformer("all-MiniLM-L6-v2")

def split_sentences(text):
    parts = []
    for line in text.split("\n"):
        for chunk in line.split("."):
            chunk = chunk.strip()
            if len(chunk) > 3:
                parts.append(chunk)
    return parts

def semantic_score(resume_text, jd):
    skills = jd["must_have"] + jd.get("nice_to_have", [])
    if not skills:
        return 0.0

    sentences = split_sentences(resume_text)
    if not sentences:
        return 0.0

    model = get_model()
    skill_emb = model.encode(skills, convert_to_tensor=True)
    sent_emb = model.encode(sentences, convert_to_tensor=True)

    sims = util.cos_sim(skill_emb, sent_emb)
    best_per_skill = sims.max(dim=1).values
    return float(best_per_skill.mean())

if __name__ == "__main__":
    jd = {
        "must_have": ["Node.js", "MongoDB", "REST APIs", "React"],
        "nice_to_have": ["Docker", "AWS"]
    }
    strong = "Built REST APIs with Express and MongoDB. Developed React frontends. Deployed with Docker on AWS."
    weak = "Experienced in graphic design and video editing. Familiar with Photoshop and Premiere."

    print("Strong resume score:", round(semantic_score(strong, jd), 3))
    print("Weak resume score:  ", round(semantic_score(weak, jd), 3))
