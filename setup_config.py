# setup_config.py

import os
from google.colab import drive
from google.oauth2 import service_account
from google.cloud import aiplatform as vertexai 

def initialize_environment(project_id: str):
    """
    Initializes the environment using a Service Account Key from Google Drive.
    This avoids the need for interactive login on every restart.
    """
    print("--- 🚀 Starting Headless Authentication ---")

    # 1. Mount Google Drive
    # This will ask for permission ONCE per session to read your files.
    if not os.path.exists('/content/drive'):
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
    
    # 2. Define Path to Key
    # Make sure your file is exactly here in your Drive!
    key_path = '/content/drive/MyDrive/Capstone_Keys/service-account.json'
    
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"❌ Could not find key file at: {key_path}. Please check your Drive folders.")

    # 3. Authenticate with the Service Account
    print(f"Loading credentials from: {key_path}")
    credentials = service_account.Credentials.from_service_account_file(key_path)
    
    # 4. Set Environment Variables
    os.environ["GCP_PROJECT_ID"] = project_id
    
    # 5. Initialize Vertex AI SDK
    try:
        vertexai.init(
            project=project_id,
            location="us-central1",
            credentials=credentials
        )
        print("==========================================================")
        print("✅ Environment successfully initialized via Service Account.")
        print(f"Project ID: {vertexai.initializer.global_config.project}")
        print(f"Service Account: {credentials.service_account_email}")
        print("==========================================================")
        
    except Exception as e:
        print(f"⚠️ ERROR: Vertex AI initialization failed: {e}")
        
    # Return modules to maintain compatibility with your notebook
    # We return 'None' for auth since we aren't using colab.auth anymore
    return None, os, vertexai