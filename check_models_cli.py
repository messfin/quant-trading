import google.generativeai as genai
import tomllib
import os

def list_models():
    secrets_path = ".streamlit/secrets.toml"
    if not os.path.exists(secrets_path):
        print("secrets.toml not found")
        return
        
    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
    
    api_key = secrets.get("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY missing in secrets.toml")
        return
    
    genai.configure(api_key=api_key)
    try:
        print("Available models:")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_models()
