def get_skill_summary(result):
    """
    Separate matched and missing skills for a candidate.
    """

    matched = []
    missing = []

    for skill, data in result["skill_coverage"].items():

        if data["present"]:
            matched.append({
                "skill": skill,
                "evidence": data["evidence"]
            })
        else:
            missing.append(skill)

    return {
        "matched": matched,
        "missing": missing
    }