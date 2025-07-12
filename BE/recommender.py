def recommend_trials_for_patient(patient_id):
    # Load patient and trial data (mocked here)
    patient = get_patient_by_id(patient_id)
    trials = get_all_trials()

    recommendations = []
    for trial in trials:
        score = compute_match_score(patient, trial)
        if score >= 0.95:
            recommendations.append({
                "trialId": trial["id"],
                "score": score
            })

    return {
        "patientId": patient_id,
        "recommendations": recommendations
    }
