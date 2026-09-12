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