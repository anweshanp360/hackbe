import flask
from flask import request, jsonify
from flask_cors import CORS
import requests

from matching_logic import match_trials  # ✅ Import the logic

app = flask.Flask(__name__)
CORS(app)

# In-memory store for frontend POST data
posted_data_store = []

# Node.js endpoints
NODE_TRIALS_API_URL = "http://localhost:3000/api/trials"
NODE_STORE_MATCHES_API_URL = "http://localhost:3000/api/matches"

@app.route('/', methods=['GET'])
def home():
    return "Python Matching Logic API is running!"

@app.route('/data', methods=['POST'])
def post_data():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    posted_data_store.append(data)
    print(f"Received and stored data: {data}")
    return jsonify({"message": "Data received and stored successfully", "data": data}), 201

@app.route('/match', methods=['GET'])
def get_matches():
    try:
        response = requests.get(NODE_TRIALS_API_URL)
        response.raise_for_status()
        trials_data = response.json()
        print(f"Fetched trials data from Node.js: {trials_data}")
    except requests.exceptions.ConnectionError:
        return jsonify({"error": f"Could not connect to Node.js trials API at {NODE_TRIALS_API_URL}."}), 500
    except requests.exceptions.Timeout:
        return jsonify({"error": f"Timeout while connecting to Node.js trials API."}), 500
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error fetching trials data: {e}"}), 500
    except ValueError:
        return jsonify({"error": "Failed to decode JSON from Node.js API response."}), 500

    if not trials_data:
        return jsonify({"message": "No trial data received from Node.js API."}), 200
    if not posted_data_store:
        return jsonify({"message": "No patient data posted yet."}), 200

    # ✅ Perform matching
    matching_results = match_trials(posted_data_store, trials_data)

    if not matching_results:
        return jsonify({"message": "No matches found."}), 200

    # ✅ Send results to Node.js for storage
    try:
        store_response = requests.post(NODE_STORE_MATCHES_API_URL, json=matching_results)
        store_response.raise_for_status()
        print("Successfully stored matching results in Node.js.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to store match results in Node.js: {e}")

    return jsonify(matching_results), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
