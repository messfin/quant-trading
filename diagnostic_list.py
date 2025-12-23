import google.generativeai as genai
import sys

def list_models_for_key(api_key):
    genai.configure(api_key=api_key)
    try:
        print("Available models:")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_models_for_key("AIzaSyCBl1GzDVIq4E_ugg7dyU4idI4-ZeQWock")
