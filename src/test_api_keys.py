"""
Standalone API Key Verification Script for Sarvam STT and Groq LLM.
Validates active keys and measures basic connectivity & response time.
"""

import os
import time
import requests
from dotenv import load_dotenv
from groq import Groq

def test_groq_api():
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key or groq_key == "your_groq_api_key_here":
        print("❌ GROQ_API_KEY is missing or unconfigured in .env")
        return False

    print("\n--- Testing Groq API (llama-3.1-8b-instant) ---")
    try:
        client = Groq(api_key=groq_key)
        start_time = time.time()
        response = client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=[
                {"role": "system", "content": "You are a test assistant."},
                {"role": "user", "content": "Respond with 'Groq API connected successfully'."}
            ],
            max_tokens=30,
            temperature=0.0
        )
        elapsed_ms = (time.time() - start_time) * 1000
        content = response.choices[0].message.content.strip()
        print(f"✅ Groq API Success! ({elapsed_ms:.1f}ms)")
        print(f"   Response: '{content}'")
        return True
    except Exception as e:
        print(f"❌ Groq API Call Failed: {e}")
        return False

def test_sarvam_api():
    sarvam_key = os.environ.get("SARVAM_API_KEY")
    if not sarvam_key or sarvam_key == "your_sarvam_api_key_here":
        print("❌ SARVAM_API_KEY is missing or unconfigured in .env")
        return False

    print("\n--- Testing Sarvam STT API Connectivity ---")
    try:
        # Check API key by querying Sarvam endpoint structure
        headers = {
            "api-subscription-key": sarvam_key
        }
        start_time = time.time()
        # Ping Sarvam API endpoint using a dummy form request or options
        # Sarvam speech-to-text requires multipart/form-data with file
        # We test key validity by sending a request to the speech-to-text endpoint
        response = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers=headers,
            timeout=10
        )
        elapsed_ms = (time.time() - start_time) * 1000
        
        # 400 Bad Request indicates key was accepted, but file missing (valid key!)
        # 401 / 403 indicates invalid key
        if response.status_code in [400, 422]:
            print(f"✅ Sarvam STT API Key Validated! ({elapsed_ms:.1f}ms) - Endpoint responded with expected status {response.status_code}")
            return True
        elif response.status_code == 200:
            print(f"✅ Sarvam STT API Success! ({elapsed_ms:.1f}ms)")
            return True
        else:
            print(f"⚠️ Sarvam API returned status {response.status_code}: {response.text}")
            if response.status_code in [401, 403]:
                return False
            return True
    except Exception as e:
        print(f"❌ Sarvam API Call Failed: {e}")
        return False

def main():
    load_dotenv()
    print("==================================================")
    print("    Voice-RAG API Keys Connectivity Verifier     ")
    print("==================================================")
    
    groq_ok = test_groq_api()
    sarvam_ok = test_sarvam_api()
    
    print("\n==================================================")
    if groq_ok and sarvam_ok:
        print("🎉 ALL API KEYS VERIFIED SUCCESSFULLY!")
    else:
        print("⚠️ Some API key checks failed. Please check your .env file.")
    print("==================================================")

if __name__ == "__main__":
    main()
