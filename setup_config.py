import os
import json
from google.colab import userdata
from google.oauth2 import service_account
from google.cloud import aiplatform as vertexai 

def initialize_environment(project_id: str):
    print("--- 🚀 Starting Cloud-Native Authentication ---")
    
    try:
        # 1. Get the JSON string from Colab Secrets
        key_json = userdata.get('GCP_CREDENTIALS')
        
        # 2. Convert string to dictionary
        key_info = json.loads(key_json)
        
        # 3. Create Credentials object directly from info
        credentials = service_account.Credentials.from_service_account_info(key_info)
        
        # 4. Initialize Vertex AI
        os.environ["GCP_PROJECT_ID"] = project_id
        vertexai.init(
            project=project_id,
            location="us-central1",
            credentials=credentials
        )
        
        print("✅ Success! Authenticated using Colab Secrets.")
        print(f"Service Account: {credentials.service_account_email}")
        
    except Exception as e:
        print(f"❌ Auth Failed: {e}")
        print("Did you add 'GCP_CREDENTIALS' to the Secrets (🔑) tab on the left?")
        
    return None, os, vertexai
