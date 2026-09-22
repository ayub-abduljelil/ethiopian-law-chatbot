import os
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

candidates = [
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

for model in candidates:
    try:
        r = client.models.generate_content(
            model=model,
            contents="Say OK.",
            config=types.GenerateContentConfig(max_output_tokens=5),
        )
        print(f"  OK    {model}  -> {r.text.strip()}")
    except Exception as e:
        err = str(e)[:80]
        print(f"  FAIL  {model}  -> {err}")
