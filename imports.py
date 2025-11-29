# imports.py
import os
from dotenv import load_dotenv

import google.generativeai as genai
from google import adk
from pydantic import BaseModel, Field

# Load variables from .env if it exists
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError(
        "GOOGLE_API_KEY not found. Set it in a .env file or as an environment variable."
    )

# Configure Gemini (Google AI Studio / Developer API)
genai.configure(api_key=api_key)

print("✅ imports.py: Authenticated with Google AI Studio and loaded ADK imports.")