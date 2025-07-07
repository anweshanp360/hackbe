from flask import Flask, jsonify, render_template, request
import pandas as pd
import numpy as np
import os
import io
import requests # Import the requests library

# Import all necessary functions from matching_logic.py
from matching_logic import (
    # Removed load_data as it will be replaced by API calls
    match_criteria,                  # Still needed for generating ML training labels
    create_ml_features_for_pair,     # For creating features for ML prediction
    generate_ml_training_data,       # For preparing the dataset to train the ML model
    train_matching_model,            # For training the ML model
    calculate_match_score,           # Import to compare with rule-based score
    CRITERIA_WEIGHTS,                # Import for max_possible_score
    x_predict                        # <--- IMPORTANT: Import x_predict here
)

# --- 1. Initialize Flask app ---
app = Flask(__name__)

# --- Global variables to store loaded data and the trained ML model ---
patients_df = pd.DataFrame()
trials_df = pd.DataFrame()
ml_model = None                      # This will hold your trained scikit-learn model
ml_feature_columns = []              # This will store the ordered list of feature names for prediction

# --- GLOBAL CONFIGURATION FOR FINAL MATCHING ---
FINAL_ML_MATCH_PROBABILITY_THRESHOLD = 0.95 # <--- YOUR DESIRED 95% THRESHOLD

# --- Node.js API Endpoints (UPDATE THESE TO YOUR ACTUAL ENDPOINTS) ---
NODE_API_BASE_URL = "http://localhost:3000" # Or your actual Node.js server address
PATIENTS_API_ENDPOINT = f"{NODE_API_BASE_URL}/patients" # Example endpoint
TRIALS_API_ENDPOINT = f"{NODE_API_BASE_URL}/trials"   # Example endpoint

# --- New functions to fetch data from Node.js APIs ---
def fetch_patients_from_api():
    print(f"Attempting to fetch patients from: {PATIENTS_API_ENDPOINT}", flush=True)
    try:
        response = requests.get(PATIENTS_API_ENDPOINT, timeout=10) # 10-second timeout
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        patient_data = response.json()
        if not patient_data:
            print("Warning: Patient API returned empty data.", flush=True)
            return pd.DataFrame()
        
        df = pd.DataFrame.from_records(patient_data)
        # Ensure 'diagnosis_code' and 'current_treatments' are treated as lists
        if 'diagnosis_code' in df.columns:
            df['diagnosis_code'] = df['diagnosis_code'].apply(lambda x: [x] if not isinstance(x, list) and x is not None else x)
        if 'current_treatments' in df.columns:
            df['current_treatments'] = df['current_treatments'].apply(lambda x: [x] if not isinstance(x, list) and x is not None else x)
        print(f"Successfully fetched {len(df)} patients from API.", flush=True)
        return df
    except requests.exceptions.Timeout:
        print(f"Error: Timeout fetching patients from {PATIENTS_API_ENDPOINT}", flush=True)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching patients from {PATIENTS_API_ENDPOINT}: {e}", flush=True)
    return pd.DataFrame()

def fetch_trials_from_api():
    print(f"Attempting to fetch trials from: {TRIALS_API_ENDPOINT}", flush=True)
    try:
        response = requests.get(TRIALS_API_ENDPOINT, timeout=10) # 10-second timeout
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        trial_data = response.json()
        if not trial_data:
            print("Warning: Trial API returned empty data.", flush=True)
            return pd.DataFrame()
        
        df = pd.DataFrame.from_records(trial_data)
        print(f"Successfully fetched {len(df)} trials from API.", flush=True)
        return df
    except requests.exceptions.Timeout:
        print(f"Error: Timeout fetching trials from {TRIALS_API_ENDPOINT}", flush=True)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching trials from {TRIALS_API_ENDPOINT}: {e}", flush=True)
    return pd.DataFrame()

# --- 2. Integrate data loading (from APIs) and ML model training when the app starts ---
with app.app_context(): # Use app_context for operations that should run once
    try:
        patients_df = fetch_patients_from_api()
        trials_df = fetch_trials_from_api()
        
        print("Data fetch process completed!")
        print(f"Fetched {len(patients_df)} patients and {len(trials_df)} trials.")

        # Check if dataframes are actually populated after fetching
        if patients_df.empty or trials_df.empty:
            print("CRITICAL ERROR: One or both dataframes are empty after API calls. ML model training will be skipped.")
            pass # Keep ml_model as None which it is initialized to.
        else:
            # Generate the ML training data and train the model
            X_full, y_full, ml_feature_columns = generate_ml_training_data(patients_df, trials_df)
            print(f"DEBUG: X_full shape: {X_full.shape}", flush=True)
            print(f"DEBUG: y_full value counts:\n{y_full.value_counts()}", flush=True)
            print(f"DEBUG: Training data label distribution: Matches={y_full.sum()}, Non-Matches={len(y_full) - y_full.sum()}", flush=True)

            if y_full.sum() == 0 or y_full.nunique() < 2:
                print("CRITICAL WARNING: No positive match examples (y=1) or only one class in training data. ML model will likely predict no matches or fail to train properly.")
                ml_model = None # Ensure model is None if training is not viable
            else:
                ml_model = train_matching_model(X_full, y_full)

            if ml_model is not None:
                print("ML Model is ready for predictions.")
                print(f"DEBUG: ML Model trained with features: {ml_feature_columns}")
                print(f"DEBUG: ML Model classes: {ml_model.classes_}")
            else:
                print("CRITICAL ERROR: ML Model could not be trained (likely due to single class in training data).")

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to fetch data from APIs or train ML model at app startup: {e}", flush=True)
        # Ensure these are explicitly reset to empty/None on error
        patients_df = pd.DataFrame()
        trials_df = pd.DataFrame()
        ml_model = None
        ml_feature_columns = []

# --- 3. Define basic routes ---

@app.route('/')
def index():
    return render_template('index.html') # Assuming you have an index.html in a 'templates' folder

@app.route('/api/patients')
def get_patients():
    global patients_df # Declare as global to ensure we're referencing the module-level variable
    if not patients_df.empty:
        # Use .copy() to avoid SettingWithCopyWarning if you modify it later
        clean_data = patients_df.where(pd.notnull(patients_df), None).copy()
        return jsonify(clean_data.to_dict(orient='records'))
    else:
        print("Warning: patients_df is empty when /api/patients was requested.", flush=True)
        return jsonify({"error": "No patient data available"}), 500

@app.route('/api/trials')
def get_trials():
    global trials_df # Declare as global
    if not trials_df.empty:
        clean_data = trials_df.where(pd.notnull(trials_df), None).copy()
        return jsonify(clean_data.to_dict(orient='records'))
    else:
        print("Warning: trials_df is empty when /api/trials was requested.", flush=True)
        return jsonify({"error": "No trial data available"}), 500

def safe_value(val):
    # This helper function needs to handle list-like inputs as well if applicable
    if isinstance(val, list):
        return [None if pd.isna(item) or item is np.nan else item for item in val]
    return None if pd.isna(val) or val is np.nan else val

# Core Matching API Endpoint to use ML model
@app.route('/api/match/<trial_id>')
def match_patients_to_trial(trial_id):
    # Declare globals for modification if needed, though mostly for access here
    global patients_df, trials_df, ml_model, ml_feature_columns, FINAL_ML_MATCH_PROBABILITY_THRESHOLD

    if patients_df.empty or trials_df.empty:
        print(f"DEBUG: System not ready. Patients empty: {patients_df.empty}, Trials empty: {trials_df.empty}", flush=True)
        return jsonify({"error": "System not ready. Data not loaded correctly. Check server logs for details."}), 500
    
    if ml_model is None or not ml_feature_columns:
        print(f"DEBUG: ML Model not ready. ML Model None: {ml_model is None}, Features empty: {not ml_feature_columns}", flush=True)
        return jsonify({"error": "System not ready. ML model not trained or feature columns missing. Check server logs for details."}), 500


    trial_row = trials_df[trials_df['trial_id'] == trial_id]

    if trial_row.empty:
        print(f"DEBUG: Trial with ID '{trial_id}' not found.", flush=True)
        return jsonify({"error": f"Trial with ID '{trial_id}' not found."}), 404

    trial_data = trial_row.iloc[0].to_dict() # Convert to dict for consistency with patient_data

    print(f"DEBUG: Starting ML match for Trial '{trial_id}' ('{trial_data.get('trial_name', 'N/A')}') with ML probability threshold {FINAL_ML_MATCH_PROBABILITY_THRESHOLD}", flush=True)
    print(f"DEBUG: Number of patients to check: {len(patients_df)}", flush=True)

    matched_patients = []
    max_possible_rule_score = sum(CRITERIA_WEIGHTS.values())


    # Loop through each patient and apply the ML matching logic
    for index, patient_data_series in patients_df.iterrows():
        patient_data = patient_data_series.to_dict() # Convert to dict

        try:
            # Use the x_predict function from matching_logic.py
            # This handles feature creation, scaling, and getting the probability correctly.
            match_probability = x_predict(ml_model, patient_data, trial_data, ml_feature_columns)

        except Exception as e:
            print(f"ERROR: Failed to predict for Patient {patient_data.get('patient_id', 'Unknown')} and Trial {trial_id} using x_predict: {e}", flush=True)
            continue

        # --- DEBUGGING OUTPUT: Print probabilities for all patients ---
        rule_based_score, individual_scores = calculate_match_score(patient_data, trial_data)
        rule_based_percentage = (rule_based_score / max_possible_rule_score) * 100 if max_possible_rule_score > 0 else 0.0
        print(f"DEBUG: Patient {patient_data['patient_id']} - ML Prob: {match_probability:.4f}, Rule-Based %: {rule_based_percentage:.2f}%", flush=True)

        # Decide if it's a match based on the ML probability threshold
        if match_probability >= FINAL_ML_MATCH_PROBABILITY_THRESHOLD:
            print(f"DEBUG: Patient {patient_data['patient_id']} ML MATCHED (Prob: {match_probability:.4f}, Rule-Based %: {rule_based_percentage:.2f}%)", flush=True)
            
            # Ensure proper handling of list-like columns for jsonify
            diagnosis_code = patient_data.get('diagnosis_code', [])
            current_treatments = patient_data.get('current_treatments', [])
            
            patient_info = {
                'patient_id': safe_value(patient_data['patient_id']),
                'match_probability': round(match_probability, 4),
                'rule_based_match_percentage': round(rule_based_percentage, 2), # Add rule-based score for comparison
                'age': safe_value(patient_data.get('age')),
                'gender': safe_value(patient_data.get('gender')),
                'diagnosis_code': safe_value(diagnosis_code), # Already a list from load_data
                'location_city': safe_value(patient_data.get('location_city')),
                'biomarker_status': safe_value(patient_data.get('biomarker_status')),
                'current_treatments': safe_value(current_treatments), # Already a list from load_data
            }
            matched_patients.append(patient_info)
            matched_patients = sorted(matched_patients, key=lambda x: x['match_probability'], reverse=True)

    try:
        max_patients = int(trial_data.get('max_patients', 100))
    except (ValueError, TypeError):
        print("WARNING: Invalid or missing max_patients field. Defaulting to 100.")
        max_patients = 100

    matched_patients = matched_patients[:max_patients]  # Limit number of matched patients

    print(f"DEBUG: Finished ML matching for Trial '{trial_id}'. Found {len(matched_patients)} matches (capped at max_patients={max_patients}) above {FINAL_ML_MATCH_PROBABILITY_THRESHOLD*100}%.", flush=True)

    return jsonify({
        'trial_id': trial_id,
        'trial_name': trial_data.get('trial_name', 'N/A'),
        'matched_patients': matched_patients,
        'count': len(matched_patients),
        'max_patients': max_patients,
        'match_threshold_applied': FINAL_ML_MATCH_PROBABILITY_THRESHOLD
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)