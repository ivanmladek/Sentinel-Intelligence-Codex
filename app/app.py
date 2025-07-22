import os
import json
import requests
from flask import Flask, request, jsonify
from google.cloud import secretmanager

app = Flask(__name__)

# --- CONFIGURATION ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
REGION = os.environ.get("GCP_REGION")
VERTEX_ENDPOINT_ID = os.environ.get("VERTEX_ENDPOINT_ID")
# The full URL for the Vertex AI Endpoint prediction
PREDICT_URL = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{VERTEX_ENDPOINT_ID}:predict"

def get_gcp_token():
    # Use requests to get the access token from the metadata server
    # This is the standard way to get credentials on GCP compute services
    metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
    response = requests.get(metadata_url, headers={"Metadata-Flavor": "Google"})
    response.raise_for_status()
    return response.json()["access_token"]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Invalid input, 'prompt' field is required."}), 400

    prompt = data['prompt']
    access_token = get_gcp_token()

    # The vLLM server expects a payload in OpenAI's format
    payload = {
        "instances": [
            {
                "prompt": prompt,
                "max_tokens": 256,
                "temperature": 0.7,
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(PREDICT_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise an exception for bad status codes
        # The actual completion is nested inside the response
        completion = response.json()["predictions"][0]
        return jsonify({"response": completion})
    except Exception as e:
        return jsonify({"error": f"Failed to call Vertex AI Endpoint: {str(e)}" , "details": response.text}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
