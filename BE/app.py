import flask
from flask import request, jsonify
from flask_cors import CORS
import requests # Import the requests library for making HTTP requests

# Initialize the Flask application
app = flask.Flask(__name__)
# Enable CORS for all routes, allowing your frontend to make requests
CORS(app)

# In-memory storage to simulate a database for posted data
# In a real application, this would be replaced with a database connection
# Example structure: [{'id': 'user123', 'value': 'some_data', 'timestamp': '...'}, ...]
posted_data_store = []

# Define the Node.js API endpoint for trials data
NODE_TRIALS_API_URL = "http://localhost:3000/api/trials"

@app.route('/', methods=['GET'])
def home():
    """
    Home endpoint to confirm the API is running.
    """
    return "Python Matching Logic API is running!"

@app.route('/data', methods=['POST'])
def post_data():
    """
    Endpoint to receive data from the frontend and store it.
    The frontend should send JSON data in the request body.
    Example JSON: {"id": "user_input_1", "name": "Product A", "value": 150}
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Add a timestamp or unique ID if needed for real database scenarios
    # For this example, we'll just append the data as is.
    posted_data_store.append(data)
    print(f"Received and stored data: {data}") # For debugging

    return jsonify({"message": "Data received and stored successfully", "data": data}), 201

@app.route('/match', methods=['GET'])
def get_matches():
    """
    Endpoint to perform the matching logic.
    It first fetches trial data from the Node.js API, then compares
    the data stored from the frontend with the fetched trial data.
    Matching criteria: Finds trials where the 'name' in the posted data
    matches the 'name' in the trials data, and the 'value' in the posted data
    is greater than or equal to the 'min_value' in the trial data.
    """
    # 1. Fetch trials data from Node.js API
    try:
        response = requests.get(NODE_TRIALS_API_URL)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        trials_data = response.json()
        print(f"Fetched trials data from Node.js: {trials_data}") # For debugging
    except requests.exceptions.ConnectionError:
        return jsonify({"error": f"Could not connect to Node.js trials API at {NODE_TRIALS_API_URL}. Make sure it's running."}), 500
    except requests.exceptions.Timeout:
        return jsonify({"error": f"Timeout while connecting to Node.js trials API at {NODE_TRIALS_API_URL}."}), 500
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error fetching trials data from Node.js: {e}"}), 500
    except ValueError: # Catches JSON decoding errors
        return jsonify({"error": "Failed to decode JSON from Node.js trials API response."}), 500

    if not trials_data:
        return jsonify({"message": "No trial data received from Node.js API to perform matching."}), 200

    if not posted_data_store:
        return jsonify({"message": "No data has been posted yet to perform matching."}), 200

    matching_results = []

    # Iterate through each piece of data posted from the frontend
    for posted_item in posted_data_store:
        posted_name = posted_item.get('name')
        posted_value = posted_item.get('value')

        if posted_name is None or posted_value is None:
            # Skip items that don't have the required fields for matching
            print(f"Skipping malformed posted item: {posted_item}")
            continue

        item_matches = []
        # Compare with each trial fetched from Node.js
        for trial in trials_data:
            trial_name = trial.get('name')
            trial_min_value = trial.get('min_value')

            # Simple matching logic:
            # Match if names are identical AND posted_value meets or exceeds trial's min_value
            if (trial_name and posted_name.lower() == trial_name.lower() and
                trial_min_value is not None and posted_value >= trial_min_value):
                item_matches.append(trial)

        if item_matches:
            matching_results.append({
                "posted_data": posted_item,
                "matches": item_matches 
            })

    if not matching_results:
        return jsonify({"message": "No matches found based on the current data and trials."}), 200
    else:
        return jsonify(matching_results), 200

# Run the Flask app
# This will run on http://127.0.0.1:5000/ by default
if __name__ == '__main__':
    app.run(debug=True, port=5000)
