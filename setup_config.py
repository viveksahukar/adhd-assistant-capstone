from google.colab import auth
import os
from google.cloud import aiplatform as vertexai 
from google.auth import default as auth_default

def initialize_environment(project_id: str):
    """
    Initializes all necessary SDKs and sets environment variables for the project.
    """
    # 1. Set Project ID and Environment Variable
    PROJECT_ID = project_id
    os.environ["GCP_PROJECT_ID"] = PROJECT_ID
    
    # 2. Authenticate the user (uncomment if needed)
    # auth.authenticate_user(project_id=PROJECT_ID)
    
    # 3. Initialize the Vertex AI SDK (The Core Connection)
    try:
        vertexai.init(project=PROJECT_ID)
        
        # 4. Optional: Print confirmation details
        credentials, project = auth_default()
        print("==========================================================")
        print("✅ Environment successfully initialized and authenticated.")
        print(f"Project ID: {vertexai.initializer.global_config.project}")
        print(f"Credentials loaded for user: {credentials.service_account_email or 'User Account'}")
        print("==========================================================")
        
    except Exception as e:
        print(f"⚠️ ERROR: Vertex AI initialization failed: {e}")
        print("Please ensure you are authenticated (run auth.authenticate_user() manually) and the Project ID is correct.")
        
    # Return modules needed for direct coding/scripting
    return auth, os, vertexai