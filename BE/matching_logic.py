import requests
import pyodbc

def match_trials(patients, trials):
    conn = pyodbc.connect(
        r'DRIVER={ODBC Driver 17 for SQL Server};'
        r'SERVER=DESKTOP-3LI0PBS,1433;'
        r'DATABASE=MyAppDB;'
        r'UID=trialAppUserNew;'
        r'PWD=Password123!;'
        r'TrustServerCertificate=Yes;'
    )
    cursor = conn.cursor()

    latest_patient = patients[-1]

    def match_score(patient, trial):
        score = 0
        total = 5

        if trial['min_age'] <= patient['age'] <= trial['max_age']:
            score += 1
        if trial['gender_requirement'].lower() == 'any' or trial['gender_requirement'].lower() == patient['gender'].lower():
            score += 1
        if patient['symptom_duration'] >= trial['min_symptom_duration']:
            score += 1
        if trial['requires_muscle_weakness'] == patient['muscle_weakness']:
            score += 1
        if 'disease_type' in trial and trial['disease_type'].lower() == patient.get('disease_type', '').lower():
            score += 1

        return (score / total) * 100

    matched_trials = []
    factors = []

    for trial in trials:
        score = match_score(latest_patient, trial)
        if score >= 95:
            matched_trials.append(trial)
            factors.append(f"TrialID {trial['trial_id']} = {score:.1f}%")

    result = {}
    if matched_trials:
        trial_ids = ",".join(str(t['trial_id']) for t in matched_trials)
        trial_names = ",".join(t['trial_name'] for t in matched_trials)
        matching_factors = "; ".join(factors)

        cursor.execute("""
            INSERT INTO MatchedPatients (patientId, patientName, trialIds, trialNames, matchingFactors)
            VALUES (?, ?, ?, ?, ?)
        """, latest_patient['patient_id'], latest_patient['patient_name'], trial_ids, trial_names, matching_factors)

        result = {
            "patientId": latest_patient['patient_id'],
            "patientName": latest_patient['patient_name'],
            "matchedTrials": matched_trials,
            "matchingFactors": matching_factors
        }

    else:
        cursor.execute("""
            INSERT INTO UnmatchedPatients (patientId, patientName, reason)
            VALUES (?, ?, ?)
        """, latest_patient['patient_id'], latest_patient['patient_name'], "No match ≥ 95% found.")

        result = {
            "patientId": latest_patient['patient_id'],
            "patientName": latest_patient['patient_name'],
            "matchedTrials": [],
            "reason": "No match ≥ 95% found."
        }

    conn.commit()
    cursor.close()
    conn.close()

    return result
