import google.generativeai as genai
import streamlit as st
import os

def list_models():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        print("API Key missing in secrets.toml")
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
