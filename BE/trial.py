import requests

# Call the API
patients_api = requests.get("http://localhost:3000/api/patients")

# Check status code
print("Status Code:", patients_api.status_code)

# Check raw response text
print("Raw Response Text:", patients_api.text)

# Try converting to JSON
try:
    patients_response = patients_api.json()
    print("Parsed JSON:", patients_response)
except Exception as e:
    print("Error parsing JSON:", e)
    patients_response = None
