# fusion.py

from backend.ranking import normalize_scores, rank_candidates

def fuse_and_rank(candidates, w_semantic=0.5, w_keyword=0.5):
    sem = normalize_scores([c["semantic_score"] for c in candidates])
    kw = normalize_scores([c["keyword_score"] for c in candidates])

    for c, s, k in zip(candidates, sem, kw):
        c["raw_semantic_score"] = c["semantic_score"]
        c["raw_keyword_score"] = c["keyword_score"]
        c["semantic_score"] = round(s, 3)
        c["keyword_score"] = round(k, 3)
        c["final_score"] = round(w_semantic * s + w_keyword * k, 3)

    return rank_candidates(candidates)

if __name__ == "__main__":
    mock = [
        {"candidate_id": "dev_strong",  "semantic_score": 0.56, "keyword_score": 0.80},
        {"candidate_id": "dev_medium",  "semantic_score": 0.40, "keyword_score": 0.45},
        {"candidate_id": "design_weak", "semantic_score": 0.14, "keyword_score": 0.05},
    ]
    ranked = fuse_and_rank(mock)
    for i, c in enumerate(ranked, 1):
        print(f"{i}. {c['candidate_id']:15} final={c['final_score']} "
              f"(sem={c['semantic_score']}, kw={c['keyword_score']})")
