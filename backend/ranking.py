def normalize_scores(scores):
    """Scale scores to the common 0-1 range used for final ranking."""
    if not scores:
        return []

    low, high = min(scores), max(scores)
    if high - low < 1e-9:
        return [0.5] * len(scores)
    return [(score - low) / (high - low) for score in scores]


def rank_candidates(results):
    """
    Sort candidates from highest final score to lowest.
    """

    return sorted(
        results,
        key=lambda candidate: candidate["final_score"],
        reverse=True
    )


def get_top_candidates(results, count=3):
    """
    Return the top N candidates.
    """

    ranked = rank_candidates(results)

    return ranked[:count]
