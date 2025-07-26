import re
import requests
from config import LLM_MODEL, GEMINI_API_KEY, GEMINI_API_URL

MEDICAL_SYNONYMS = {
    "knee surgery": ["orthopedic procedure", "joint surgery", "arthroscopy"],
    "heart attack": ["myocardial infarction", "cardiac arrest", "coronary event"],
    "cancer": ["malignancy", "tumor", "carcinoma"],
    "hospitalization": ["inpatient care", "admission", "hospital stay"],
    "maternity": ["pregnancy", "childbirth", "delivery"],
    "surgery": ["procedure", "operation", "intervention"],
    "knee": ["joint", "articular", "patellar"],
    # Add more as needed
}

llm = None

def gemini_generate(prompt, max_tokens=64, temperature=0.3):
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature}
    }
    response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data)
    response.raise_for_status()
    result = response.json()
    text = result["candidates"][0]["content"]["parts"][0]["text"] if "candidates" in result and result["candidates"] else ""
    return text

def get_dynamic_synonyms(term):
    prompt = f"You are a medical insurance expert. List 5 synonyms or related terms for: '{term}'.\nSynonyms:"
    text = gemini_generate(prompt, max_tokens=64, temperature=0.3)
    synonyms = [t.strip() for t in text.split(',') if t.strip()]
    return synonyms

# Demographic parsing: e.g. "46M Pune" or "32F Mumbai"
def parse_demographics(query):
    match = re.search(r"(\d{1,3})([MF])\s*([A-Za-z ]+)?", query)
    if match:
        age = int(match.group(1))
        gender = match.group(2)
        location = match.group(3).strip() if match.group(3) else None
        return {"age": age, "gender": gender, "location": location}
    return {}

# Policy duration extraction: e.g. "3-month", "2 years", "90 days"
def extract_policy_duration(query):
    patterns = [
        (r"(\d+)\s*-?\s*month", lambda m: int(m.group(1)) * 30),
        (r"(\d+)\s*-?\s*year", lambda m: int(m.group(1)) * 365),
        (r"(\d+)\s*-?\s*day", lambda m: int(m.group(1))),
    ]
    for pat, func in patterns:
        match = re.search(pat, query, re.IGNORECASE)
        if match:
            return func(match)
    return None
