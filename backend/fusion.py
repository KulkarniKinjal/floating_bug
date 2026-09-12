# fusion.py

def normalize(scores):
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]

def fuse_and_rank(candidates, w_semantic=0.5, w_keyword=0.5):
    sem = normalize([c["semantic_score"] for c in candidates])
    kw = normalize([c["keyword_score"] for c in candidates])

    for c, s, k in zip(candidates, sem, kw):
        c["final_score"] = round(w_semantic * s + w_keyword * k, 3)

    return sorted(candidates, key=lambda c: c["final_score"], reverse=True)

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