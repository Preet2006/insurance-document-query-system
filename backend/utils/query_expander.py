import requests
from config import GEMINI_API_KEY, GEMINI_API_URL
import json

class QueryExpander:
    def __init__(self, llm=None):
        self.llm = llm

    def mistral_expand(self, query: str) -> dict:
        if self.llm is None:
            # Fallback: return minimal expansion if llm is not provided
            return {"demographics": {}, "expansions": [query], "inferred_intent": "", "duration_days": None}
        prompt = f'''
You are an expert insurance query understanding agent. Given the user query below, extract:
- Demographics (age, gender, location)
- Expansions (3-5 alternative phrasings or related terms)
- Inferred intent (e.g., "Check coverage under Section 4.2 (Surgical Benefits)")
- Policy duration in days (if mentioned)

Return a JSON like:
{{
  "demographics": {{"age": ..., "gender": ..., "location": ...}},
  "expansions": [...],
  "inferred_intent": "...",
  "duration_days": ...
}}

Query: "{query}"
JSON: '''
        output = self.llm(prompt, max_tokens=256)
        text = output['choices'][0]['text'] if isinstance(output, dict) and 'choices' in output else str(output)
        if text.strip().startswith("```"):
            lines = text.strip().splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {"demographics": {}, "expansions": [query], "inferred_intent": "", "duration_days": None}
        return parsed

    def gemini_expand(self, query: str) -> dict:
        prompt = f'''
You are an expert insurance query understanding agent. Given the user query below, extract:
- Demographics (age, gender, location)
- Expansions (3-5 alternative phrasings or related terms)
- Inferred intent (e.g., "Check coverage under Section 4.2 (Surgical Benefits)")
- Policy duration in days (if mentioned)

Return a JSON like:
{{
  "demographics": {{"age": ..., "gender": ..., "location": ...}},
  "expansions": [...],
  "inferred_intent": "...",
  "duration_days": ...
}}

Query: "{query}"
JSON: '''
        headers = {"Content-Type": "application/json"}
        params = {"key": GEMINI_API_KEY}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 256, "temperature": 0.3}
        }
        try:
            response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data)
            response.raise_for_status()
            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"] if "candidates" in result and result["candidates"] else ""
            if text.strip().startswith("```"):
                lines = text.strip().splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = {"demographics": {}, "expansions": [query], "inferred_intent": "", "duration_days": None}
        except Exception:
            # Fallback to Mistral
            return self.mistral_expand(query)
        return parsed

    def expand(self, query: str) -> dict:
        return self.gemini_expand(query)
