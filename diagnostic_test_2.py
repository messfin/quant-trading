import google.generativeai as genai
import sys

def test_key(api_key):
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Hello, reply with only one word.")
        print(f"Success: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_key("AIzaSyCBl1GzDVIq4E_ugg7dyU4idI4-ZeQWock")
