# ml_logic/match_trials.py
import sys
import json

def run_ml_logic(patient_data, trial_data):
    # Your ML logic goes here
    # Example: Simple matching based on age and disease
    matching_trials = []
    for trial in trial_data:
        # Implement your complex matching algorithm
        if (patient_data.get('age') >= trial.get('min_age', 0) and
            patient_data.get('age') <= trial.get('max_age', 200) and
            patient_data.get('disease') == trial.get('disease_required')):
            matching_trials.append(trial)
    return matching_trials

if __name__ == '__main__':
    # Read the JSON string from command-line arguments
    input_data_str = sys.argv[1]
    input_data = json.loads(input_data_str)

    patient = input_data['patient']
    trials = input_data['trials']

    # Perform the ML logic
    results = run_ml_logic(patient, trials)

    # Print the result as a JSON string to stdout
    print(json.dumps(results))