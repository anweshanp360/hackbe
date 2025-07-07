import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import requests # NEW: Import requests library for making HTTP calls

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

# --- Configuration for Scoring ---
CRITERIA_WEIGHTS = {
    "diagnosis_code": 0.30,
    "age": 0.20,
    "gender": 0.10,
    "biomarker_status": 0.20,
    "excluded_treatment": 0.10,
    "location": 0.10,
}

AGE_FLEXIBILITY_YEARS = 5
LOCATION_PROXIMITY_THRESHOLD_KM = 50

CITY_COORDINATES = {
    "Kolkata": (22.5726, 88.3639),
    "Delhi": (28.7041, 77.1025),
    "Mumbai": (19.0760, 72.8777),
    "Chennai": (13.0827, 80.2707),
    "Bangalore": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Any": None # 'Any' location means no geographical constraint
}

# --- NEW: Configuration for Node.js API endpoint ---
# It's best practice to use an environment variable for this in production.
# For local development, a default can be provided.
NODEJS_API_BASE_URL = os.environ.get('NODEJS_API_BASE_URL', 'http://localhost:3000')


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance

def calculate_match_score(patient, trial):
    """
    Calculates a rule-based match score between a patient and a trial based on predefined criteria.
    Returns the total score and a dictionary of individual criterion scores.
    """
    score = 0
    individual_scores = {}

    # 1. Diagnosis Code Match
    # Ensure patient_diagnosis_codes is a list. If it comes as a single string (e.g., "C50.9"),
    # or a comma-separated string, parse it into a list.
    patient_diagnosis_codes = patient.get('diagnosis_code', [])
    if isinstance(patient_diagnosis_codes, str):
        patient_diagnosis_codes = [code.strip() for code in patient_diagnosis_codes.split(',') if code.strip()]
    patient_diagnosis_codes = [code.lower() for code in patient_diagnosis_codes if code] # Normalize to lower case

    # Ensure trial_required_diagnosis_code is handled as a list if it can contain multiple
    # Assuming 'required_diagnosis_code' in trial data is a string or single item from API.
    # If it can be a list from the API, ensure it's processed as such.
    trial_required_diagnosis_code = str(trial.get('required_diagnosis_code', 'None')).strip().lower()

    # If the trial has specific required codes, check if any patient code matches one of them.
    # If trial_required_diagnosis_code is 'none', it means no specific diagnosis is required.
    if trial_required_diagnosis_code == 'none':
        individual_scores["diagnosis_code"] = 1.0 * CRITERIA_WEIGHTS["diagnosis_code"]
    elif trial_required_diagnosis_code in patient_diagnosis_codes:
        individual_scores["diagnosis_code"] = 1.0 * CRITERIA_WEIGHTS["diagnosis_code"]
    else:
        individual_scores["diagnosis_code"] = 0.0
    score += individual_scores["diagnosis_code"]

    # 2. Age Match
    patient_age = patient.get('age', 0)
    min_age = trial.get('min_age', 0)
    max_age = trial.get('max_age', 150) # Default to a wide range if not specified

    age_score = 0.0
    if min_age <= patient_age <= max_age:
        age_score = 1.0 # Perfect match if within range
    elif patient_age < min_age:
        if min_age - patient_age <= AGE_FLEXIBILITY_YEARS:
            # Linear decay from 1.0 to 0.0 over AGE_FLEXIBILITY_YEARS
            age_score = 1.0 - ((min_age - patient_age) / AGE_FLEXIBILITY_YEARS)
    elif patient_age > max_age:
        if patient_age - max_age <= AGE_FLEXIBILITY_YEARS:
            # Linear decay from 1.0 to 0.0 over AGE_FLEXIBILITY_YEARS
            age_score = 1.0 - ((patient_age - max_age) / AGE_FLEXIBILITY_YEARS)
    individual_scores["age"] = age_score * CRITERIA_WEIGHTS["age"]
    score += individual_scores["age"]

    # 3. Gender Match
    patient_gender = str(patient.get('gender', 'Unknown')).strip().lower()
    required_gender = str(trial.get('required_gender', 'Any')).strip().lower()
    if required_gender == 'any' or patient_gender == required_gender:
        individual_scores["gender"] = 1.0 * CRITERIA_WEIGHTS["gender"]
    else:
        individual_scores["gender"] = 0.0
    score += individual_scores["gender"]

    # 4. Biomarker Status Match
    patient_biomarker_status = str(patient.get('biomarker_status', 'Unknown')).strip().lower()
    required_biomarker_status = str(trial.get('required_biomarker_status', 'N/A')).strip().lower() # N/A means not required
    if required_biomarker_status == 'n/a' or patient_biomarker_status == required_biomarker_status:
        individual_scores["biomarker_status"] = 1.0 * CRITERIA_WEIGHTS["biomarker_status"]
    else:
        individual_scores["biomarker_status"] = 0.0
    score += individual_scores["biomarker_status"]

    # 5. Excluded Treatment Match (Patient should NOT have the excluded treatment)
    patient_current_treatments = patient.get('current_treatments', [])
    if isinstance(patient_current_treatments, str):
        patient_current_treatments = [treatment.strip() for treatment in patient_current_treatments.split(',') if treatment.strip()]
    patient_current_treatments = [t.lower() for t in patient_current_treatments if t]

    excluded_treatment = str(trial.get('excluded_treatment', 'None')).strip().lower() # 'None' means no excluded treatment
    if excluded_treatment == 'none':
        individual_scores["excluded_treatment"] = 1.0 * CRITERIA_WEIGHTS["excluded_treatment"] # No exclusion means full score
    elif excluded_treatment in patient_current_treatments: # Patient has excluded treatment
        individual_scores["excluded_treatment"] = 0.0
    else:
        individual_scores["excluded_treatment"] = 1.0 * CRITERIA_WEIGHTS["excluded_treatment"] # Patient does not have excluded treatment
    score += individual_scores["excluded_treatment"]

    # 6. Location Match
    patient_location_city = str(patient.get('location_city', 'Unknown')).strip()
    # Assuming 'location_cities' in trial is a list of allowed cities, or 'Any' for no restriction
    trial_location_cities = trial.get('location_cities', [])
    if isinstance(trial_location_cities, str): # Handle if it comes as a single string from DB/API
        trial_location_cities = [city.strip() for city in trial_location_cities.split(',') if city.strip()]
    trial_location_cities = [city.lower() for city in trial_location_cities if city] # Normalize to lower case

    location_score = 0.0
    if 'any' in trial_location_cities or not trial_location_cities: # Trial is open to any location or list is empty
        location_score = 1.0
    elif patient_location_city.lower() in trial_location_cities: # Exact city match
        location_score = 1.0
    else:
        # Calculate distance if coordinates are available for patient's city and any of the trial's target cities
        patient_coords = CITY_COORDINATES.get(patient_location_city)
        if patient_coords:
            min_dist = float('inf')
            for target_city in trial_location_cities:
                trial_coords = CITY_COORDINATES.get(target_city)
                if trial_coords:
                    dist = haversine_distance(patient_coords[0], patient_coords[1], trial_coords[0], trial_coords[1])
                    min_dist = min(min_dist, dist)
            
            if min_dist <= LOCATION_PROXIMITY_THRESHOLD_KM:
                # Linear decay: score is 1.0 at 0km distance, 0.0 at LOCATION_PROXIMITY_THRESHOLD_KM
                location_score = 1.0 - (min_dist / LOCATION_PROXIMITY_THRESHOLD_KM)
                location_score = max(0.0, min(1.0, location_score)) # Clamp between 0 and 1
        # If coordinates are missing for patient city or all trial cities, or distance > threshold, score remains 0.0
    individual_scores["location"] = location_score * CRITERIA_WEIGHTS["location"]
    score += individual_scores["location"]

    return score, individual_scores

def match_criteria(patient, trial):
    """
    Determines a binary match (True/False) between a patient and a trial
    for ML training labels, based on a normalized match score threshold.
    """
    score, _ = calculate_match_score(patient, trial)
    max_possible_score = sum(CRITERIA_WEIGHTS.values())
    
    # Prevent division by zero
    if max_possible_score == 0:
        match_percentage = 0.0
    else:
        match_percentage = (score / max_possible_score) * 100

    # logging.debug(f"Match % for patient-trial pair: {match_percentage:.2f}") # Use logging instead of print

    # This threshold is CRITICAL for your ML model's performance.
    # Adjust it based on the distribution of your rule-based scores.
    TRAINING_MATCH_THRESHOLD_PERCENTAGE = 40.0
    return match_percentage >= TRAINING_MATCH_THRESHOLD_PERCENTAGE

def create_ml_features_for_pair(patient_data, trial_data):
    """
    Generates a dictionary of features for a given patient-trial pair,
    to be used as input for the ML model.
    """
    features = {}

    # Robustly get values with defaults and normalize to lower case/lists
    patient_age = patient_data.get('age', 0)
    min_age = trial_data.get('min_age', 0)
    max_age = trial_data.get('max_age', 150) # Default for max_age should be high
    
    patient_gender = str(patient_data.get('gender', 'Unknown')).strip().lower() # Normalize to lower case
    trial_gender = str(trial_data.get('required_gender', 'Any')).strip().lower() # Normalize to lower case

    patient_diagnosis_codes_raw = patient_data.get('diagnosis_code', [])
    if isinstance(patient_diagnosis_codes_raw, str):
        patient_diagnosis_codes = [code.strip().lower() for code in patient_diagnosis_codes_raw.split(',') if code.strip()]
    else: # Assume it's already a list or similar
        patient_diagnosis_codes = [str(code).strip().lower() for code in patient_diagnosis_codes_raw if str(code).strip()]
    
    trial_required_diagnosis_code = str(trial_data.get('required_diagnosis_code', 'None')).strip().lower()

    patient_biomarker_status = str(patient_data.get('biomarker_status', 'Unknown')).strip().lower()
    trial_required_biomarker_status = str(trial_data.get('required_biomarker_status', 'N/A')).strip().lower()

    patient_current_treatments_raw = patient_data.get('current_treatments', [])
    if isinstance(patient_current_treatments_raw, str):
        patient_current_treatments = [treatment.strip().lower() for treatment in patient_current_treatments_raw.split(',') if treatment.strip()]
    else: # Assume it's already a list or similar
        patient_current_treatments = [str(treatment).strip().lower() for treatment in patient_current_treatments_raw if str(treatment).strip()]
    
    trial_excluded_treatment = str(trial_data.get('excluded_treatment', 'None')).strip().lower()

    patient_location_city = str(patient_data.get('location_city', 'Unknown')).strip()
    trial_target_location_cities_raw = trial_data.get('location_cities', [])
    if isinstance(trial_target_location_cities_raw, str):
        trial_target_location_cities = [city.strip().lower() for city in trial_target_location_cities_raw.split(',') if city.strip()]
    else:
        trial_target_location_cities = [str(city).strip().lower() for city in trial_target_location_cities_raw if str(city).strip()]
    

    # --- Feature Generation ---

    # Age features
    features['age_deviation_low'] = max(0, min_age - patient_age)
    features['age_deviation_high'] = max(0, patient_age - max_age)
    features['age_diff_abs'] = abs(patient_age - (min_age + max_age) / 2) if (min_age + max_age) > 0 else 0
    features['age_is_within_range'] = 1 if (min_age <= patient_age <= max_age) else 0

    age_range_midpoint = (min_age + max_age) / 2
    age_range_half_width = (max_age - min_age) / 2
    if age_range_half_width > 0:
        features['age_normalized_deviation_from_center'] = abs(patient_age - age_range_midpoint) / age_range_half_width
        features['age_proximity_score_normalized'] = 1 - min(1, features['age_normalized_deviation_from_center']) # Closer to 1 for better match
    else: # Trial has fixed age
        features['age_normalized_deviation_from_center'] = 0 if patient_age == min_age else 1
        features['age_proximity_score_normalized'] = 1 if patient_age == min_age else 0

    # Gender feature
    features['gender_match'] = 1 if (trial_gender == 'any' or patient_gender == trial_gender) else 0 # Changed to patient_gender == trial_gender for clarity
    features['gender_mismatch'] = 1 - features['gender_match'] # Explicit mismatch feature

    # Diagnosis feature
    features['diagnosis_exact_match'] = 1 if trial_required_diagnosis_code in patient_diagnosis_codes else 0
    features['num_matching_diagnosis_codes'] = 1 if trial_required_diagnosis_code != 'none' and trial_required_diagnosis_code in patient_diagnosis_codes else 0
    features['patient_diagnosis_code_count'] = len(patient_diagnosis_codes)
    features['trial_requires_specific_diagnosis'] = 1 if trial_required_diagnosis_code != 'none' else 0


    # Biomarker feature
    features['biomarker_exact_match'] = 1 if (trial_required_biomarker_status == 'n/a' or trial_required_biomarker_status == patient_biomarker_status) else 0
    features['biomarker_mismatch'] = 1 - features['biomarker_exact_match']
    # Consider one-hot encoding for biomarker status if there are more than two relevant categories
    # E.g., features['biomarker_status_positive'] = 1 if patient_biomarker_status == 'positive' else 0


    # Excluded Treatment feature
    features['excluded_treatment_conflict'] = 1 if (trial_excluded_treatment != 'none' and trial_excluded_treatment in patient_current_treatments) else 0
    features['excluded_treatment_no_conflict'] = 1 - features['excluded_treatment_conflict']
    features['patient_has_treatments'] = 1 if len(patient_current_treatments) > 0 else 0
    features['trial_has_excluded_treatment'] = 1 if trial_excluded_treatment != 'none' else 0


    # Location features
    MAX_REASONABLE_DISTANCE_KM = 5000 # A reasonable max distance for normalization, can be adjusted
    
    features['location_distance_km'] = MAX_REASONABLE_DISTANCE_KM # Default to max distance for non-matches
    features['location_is_any_or_unknown'] = 0 # Default

    if 'any' in trial_target_location_cities or not trial_target_location_cities: # Trial is open to any location
        features['location_distance_km'] = 0.0 # Treat as perfect match
        features['location_is_any_or_unknown'] = 1
    elif patient_location_city.lower() == 'unknown': # Patient location is unknown
        features['location_is_any_or_unknown'] = 1
        # features['location_distance_km'] remains MAX_REASONABLE_DISTANCE_KM
    elif patient_location_city.lower() in trial_target_location_cities: # Exact city match
        features['location_distance_km'] = 0.0
    else:
        # Calculate distance to the closest target city for the trial
        patient_coords = CITY_COORDINATES.get(patient_location_city)
        if patient_coords:
            min_dist_to_trial = float('inf')
            for target_city in trial_target_location_cities:
                trial_coords = CITY_COORDINATES.get(target_city)
                if trial_coords:
                    dist = haversine_distance(patient_coords[0], patient_coords[1], trial_coords[0], trial_coords[1])
                    min_dist_to_trial = min(min_dist_to_trial, dist)
            if min_dist_to_trial != float('inf'): # If at least one coordinate pair was found
                features['location_distance_km'] = min_dist_to_trial
            # Else, it remains MAX_REASONABLE_DISTANCE_KM if no valid coordinates for trial cities
            
    features['location_exact_city_match'] = 1 if (patient_location_city.lower() in trial_target_location_cities) else 0
    features['location_within_threshold'] = 1 if features['location_distance_km'] <= LOCATION_PROXIMITY_THRESHOLD_KM else 0
    features['location_normalized_distance_score'] = 1 - min(1, features['location_distance_km'] / MAX_REASONABLE_DISTANCE_KM)
    features['trial_is_location_agnostic'] = 1 if ('any' in trial_target_location_cities or not trial_target_location_cities) else 0

    # Interaction features (examples) - Keep these, they are good.
    features['diagnosis_age_interaction'] = features['diagnosis_exact_match'] * features['age_is_within_range']
    features['biomarker_location_interaction'] = features['biomarker_exact_match'] * features['location_within_threshold']
    features['age_gender_biomarker_match'] = features['age_is_within_range'] * features['gender_match'] * features['biomarker_exact_match']

    return features

def generate_ml_training_data(patients_df, trials_df):
    """
    Generates features (X) and labels (y) for ML training.
    """
    X_data = []
    y_labels = []
    positive_matches_count = 0

    patients_list = patients_df.to_dict(orient='records')
    trials_list = trials_df.to_dict(orient='records')

    # Get a comprehensive list of all possible feature names
    # Create one dummy pair to get all feature names
    if patients_list and trials_list:
        dummy_patient = patients_list[0]
        dummy_trial = trials_list[0]
        dummy_features = create_ml_features_for_pair(dummy_patient, dummy_trial)
        feature_cols = list(dummy_features.keys())
    else:
        logging.warning("No patient or trial data to generate dummy features for feature_cols. Returning empty dataframes.")
        return pd.DataFrame(), pd.Series(), []

    for patient in patients_list:
        for trial in trials_list:
            features = create_ml_features_for_pair(patient, trial)
            label = int(match_criteria(patient, trial)) # Convert True/False to 1/0
            
            # Ensure all features are present for every sample in the correct order
            feature_vector = [features.get(col, 0.0) for col in feature_cols] # Default to 0.0 if feature missing
            X_data.append(feature_vector)
            y_labels.append(label)

            if label == 1:
                positive_matches_count += 1

    X = pd.DataFrame(X_data, columns=feature_cols)
    y = pd.Series(y_labels)

    logging.debug("--- Generated ML Training Data Summary ---")
    logging.debug(f"Total samples generated: {len(X)}")
    logging.debug(f"Positive matches (label 1): {positive_matches_count}")
    logging.debug(f"Negative matches (label 0): {len(X) - positive_matches_count}")

    if positive_matches_count == 0:
        logging.critical("No positive matches (label 1) generated in training data. Model will only learn class 0.")
        logging.critical("ACTION: Adjust 'TRAINING_MATCH_THRESHOLD_PERCENTAGE' in 'match_criteria' or check your raw data/rule-based scoring logic.")

    logging.debug(f"y_series value counts:\n{y.value_counts().to_string()}")

    return X, y, feature_cols

def train_matching_model(X_train, y_train):
    """
    Trains a RandomForestClassifier model.
    """
    if y_train.nunique() < 2:
        logging.critical(f"Only one unique class found in 'y_train' ({y_train.unique()[0]}). Cannot train a binary classifier.")
        logging.critical("Please ensure your training data contains both positive and negative examples.")
        return None

    # Scale numerical features (important for many ML models, though RF is less sensitive)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Use class_weight='balanced' to handle imbalanced datasets
    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # Store the scaler with the model for later use during prediction
    model.scaler = scaler
    # Store feature names for consistent prediction, important when loading/saving models
    model.feature_names_in_ = X_train.columns.tolist()

    return model

def x_predict(model, patient_data, trial_data, feature_columns):
    # Ensure patient_data and trial_data are dictionaries for create_ml_features_for_pair
    # This might already be handled by app.py's .to_dict() but good for robustness
    if not isinstance(patient_data, dict):
        patient_data = patient_data.to_dict()
    if not isinstance(trial_data, dict):
        trial_data = trial_data.to_dict()

    # Create features for this specific pair
    features_dict = create_ml_features_for_pair(patient_data, trial_data)

    # Convert the feature dictionary to a Pandas DataFrame row, ensuring column order
    # It's crucial that feature_columns passed here matches the order the model was trained with.
    X_predict_single = pd.DataFrame([features_dict], columns=feature_columns)

    # Scale the features using the scaler stored on the model (if it exists)
    if hasattr(model, 'scaler') and model.scaler is not None:
        X_predict_single_scaled = model.scaler.transform(X_predict_single)
    else:
        logging.warning("Scaler not found on ML model. Prediction might be unscaled. This can lead to incorrect predictions.")
        X_predict_single_scaled = X_predict_single.values  # Convert DataFrame to numpy array

    # Predict probabilities
    probs = model.predict_proba(X_predict_single_scaled)[0]

    # Get the probability of the positive class (class 1)
    match_probability = 0.0
    if 1 in model.classes_:
        match_probability = probs[np.where(model.classes_ == 1)[0][0]]
    elif len(model.classes_) == 1 and model.classes_[0] == 0:
        match_probability = 0.0  # Only class 0 was seen in training, so probability of 1 is 0
    # If len(model.classes_) == 1 and model.classes_[0] == 1, then all training data was 1.
    # In this unlikely scenario, predict_proba might return [1.0] for class 1.
    elif len(model.classes_) == 1 and model.classes_[0] == 1:
        match_probability = 1.0 # Only class 1 was seen in training

    return match_probability


def load_data():
    """
    Fetches patient and trial data from the Node.js API endpoints.
    Handles potential network errors and basic data type conversions.
    """
    patients_url = f"{NODEJS_API_BASE_URL}/api/patients"
    trials_url = f"{NODEJS_API_BASE_URL}/api/trials"

    logging.info(f"Attempting to fetch patients from: {patients_url}")
    logging.info(f"Attempting to fetch trials from: {trials_url}")

    patients_data = []
    trials_data = []

    try:
        patients_response = requests.get(patients_url, timeout=15) # Increased timeout
        patients_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        patients_data = patients_response.json()
        logging.info(f"Successfully fetched {len(patients_data)} patients.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching patients from Node.js API at {patients_url}: {e}")
        # Return empty DataFrame if critical data fetch fails.
        # The calling app.py will handle this by skipping ML training.
        return pd.DataFrame(), pd.DataFrame()

    try:
        trials_response = requests.get(trials_url, timeout=15) # Increased timeout
        trials_response.raise_for_status()
        trials_data = trials_response.json()
        logging.info(f"Successfully fetched {len(trials_data)} trials.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching trials from Node.js API at {trials_url}: {e}")
        # Return empty DataFrame if critical data fetch fails.
        return pd.DataFrame(), pd.DataFrame()

    patients_df = pd.DataFrame(patients_data)
    trials_df = pd.DataFrame(trials_data)

    # --- Data Cleaning and Type Conversion based on typical API JSON structures ---
    # This part is crucial! Ensure your data types are correct for ML feature creation.

    # Convert numeric columns, coercing errors to NaN
    for col in ['age', 'min_age', 'max_age', 'max_patients']:
        if col in patients_df.columns:
            patients_df[col] = pd.to_numeric(patients_df[col], errors='coerce')
        if col in trials_df.columns:
            trials_df[col] = pd.to_numeric(trials_df[col], errors='coerce')

    # Handle list-like columns that might come as strings or non-list types from JSON
    # For 'diagnosis_code' in patients_df:
    if 'diagnosis_code' in patients_df.columns:
        patients_df['diagnosis_code'] = patients_df['diagnosis_code'].apply(
            lambda x: [item.strip() for item in str(x).split(',') if item.strip()] if isinstance(x, str) else 
                      ([str(item).strip() for item in x if str(item).strip()] if isinstance(x, list) else [])
        )

    # For 'current_treatments' in patients_df:
    if 'current_treatments' in patients_df.columns:
        patients_df['current_treatments'] = patients_df['current_treatments'].apply(
            lambda x: [item.strip() for item in str(x).split(',') if item.strip()] if isinstance(x, str) else 
                      ([str(item).strip() for item in x if str(item).strip()] if isinstance(x, list) else [])
        )
    
    # For 'location_cities' in trials_df:
    if 'location_cities' in trials_df.columns:
        trials_df['location_cities'] = trials_df['location_cities'].apply(
            lambda x: [item.strip() for item in str(x).split(',') if item.strip()] if isinstance(x, str) else 
                      ([str(item).strip() for item in x if str(item).strip()] if isinstance(x, list) else [])
        )

    # For 'required_diagnosis_code' in trials_df:
    # Assuming this might be a single string from Node.js, ensure it's handled consistently as such.
    # If it can be a list, adjust this part. For now, treating as single string.
    if 'required_diagnosis_code' in trials_df.columns:
        trials_df['required_diagnosis_code'] = trials_df['required_diagnosis_code'].apply(
            lambda x: str(x).strip() if pd.notna(x) else 'None'
        )
        
    # For 'excluded_treatment' in trials_df:
    # Assuming this might be a single string from Node.js, ensure it's handled consistently as such.
    if 'excluded_treatment' in trials_df.columns:
        trials_df['excluded_treatment'] = trials_df['excluded_treatment'].apply(
            lambda x: str(x).strip() if pd.notna(x) else 'None'
        )

    # Fill NaN values in object/string columns with 'Unknown' or appropriate default
    for col in ['gender', 'biomarker_status', 'location_city']:
        if col in patients_df.columns:
            patients_df[col] = patients_df[col].fillna('Unknown')
    for col in ['required_gender', 'required_biomarker_status']:
        if col in trials_df.columns:
            trials_df[col] = trials_df[col].fillna('N/A') # Or 'Any' based on your schema

    # It's good to log the first few rows after loading and cleaning for verification
    logging.debug("\nPatients DataFrame head after loading and cleaning:")
    logging.debug(patients_df.head().to_string())
    logging.debug("\nTrials DataFrame head after loading and cleaning:")
    logging.debug(trials_df.head().to_string())

    return patients_df, trials_df