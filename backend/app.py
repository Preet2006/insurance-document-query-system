import os
os.environ['FLASK_NO_COLOR'] = '1'  # Disable colored output

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
import asyncio
from config import *
import chromadb
from chromadb.utils import embedding_functions
import requests
from config import GEMINI_API_KEY, GEMINI_API_URL
import atexit
from utils.medical_terms import get_dynamic_synonyms, parse_demographics, extract_policy_duration
from utils.query_expander import QueryExpander
import re
from utils.rule_engine import RuleEngine
import json
from datetime import datetime, timezone
import cohere
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# === COHERE API KEY SETUP ===
# Paste your Cohere API key below (or load from environment/config securely)
COHERE_API_KEY = "qZmghdKw7d7YxNryMj57OsMN0jLsQSCy0c7xulRA"
co = cohere.Client(COHERE_API_KEY)

# Simple in-memory cache for recent queries
query_cache = {}
CACHE_SIZE = 100

def cache_get(key):
    return query_cache.get(key)

def cache_set(key, value):
    if len(query_cache) >= CACHE_SIZE:
        # Remove oldest entry
        oldest_key = next(iter(query_cache))
        del query_cache[oldest_key]
    query_cache[key] = value

def redact_pii(text):
    # Redact emails, phone numbers, and policy numbers (less aggressive)
    text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED_EMAIL]", text)
    text = re.sub(r"\b\d{10,}\b", "[REDACTED_PHONE]", text)
    text = re.sub(r"\b[A-Z0-9]{8,}\b", "[REDACTED_POLICY]", text)
    # Do NOT redact all capitalized words (names)
    return text

def redact_pii_in_dict(d):
    if isinstance(d, dict):
        return {k: redact_pii_in_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [redact_pii_in_dict(x) for x in d]
    elif isinstance(d, str):
        return redact_pii(d)
    else:
        return d

def audit_log(entry):
    with open("audit.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def count_tokens(text):
    # Simple approximation: 1 token ≈ 4 characters
    return len(text) // 4

def truncate_clauses_to_fit(clauses, prompt_prefix, max_tokens=2048, buffer=200):
    current_tokens = count_tokens(prompt_prefix)
    truncated_clauses = []
    for clause in clauses:
        clause_text = clause['text']
        clause_tokens = count_tokens(clause_text)
        if current_tokens + clause_tokens + buffer > max_tokens:
            break
        truncated_clauses.append(clause)
        current_tokens += clause_tokens
    return truncated_clauses

# Initialize components
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
# NOTE: The following embedding_function type warning is a linter issue, not a runtime error for your use case.
collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=str(EMBEDDING_MODEL)
    )  # type: ignore
)

SYSTEM_PROMPT = """You are an expert insurance policy analyst. Use only the context below to answer the question. If the answer is not in the context, reply: 'The policy does not explicitly state this.'\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""

def gemini_generate(prompt, max_tokens=512, temperature=0.2):
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature}
    }
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        text = result["candidates"][0]["content"]["parts"][0]["text"] if "candidates" in result and result["candidates"] else ""
    except requests.exceptions.Timeout:
        return "Gemini API request timed out."
    except Exception as e:
        return f"Gemini API error: {str(e)}"
    return text

def analyze_decision(answer, clauses_used):
    # Simple rules for demo: can be expanded with more logic
    decision = "pending"
    coverage_amount = None
    summary = "Unable to determine coverage from the provided context."
    answer_lc = answer.lower() if answer else ""
    if "not covered" in answer_lc or "excluded" in answer_lc:
        decision = "denied"
        summary = "The policy explicitly excludes this scenario."
    elif "covered" in answer_lc or "included" in answer_lc:
        decision = "approved"
        # Try to extract percentage
        import re
        match = re.search(r'(\d+\s*%)', answer_lc)
        if match:
            coverage_amount = match.group(1)
        summary = "The policy covers this scenario."
    elif "waiting period" in answer_lc:
        decision = "pending"
        summary = "A waiting period applies."
    elif "network hospital" in answer_lc:
        decision = "pending"
        summary = "Coverage depends on network hospital status."
    return decision, coverage_amount, summary

def ensure_response_schema(llm_output, clauses_used):
    # Fill in missing fields with defaults and ensure clause_refs is a list of dicts
    def format_clause_ref(clause):
        return {
            "document": clause.get("document", "Unknown"),
            "section": clause.get("section", "Unknown"),
            "text": clause.get("text", "Unknown")
        }
    clause_refs = llm_output.get("clause_refs")
    if not clause_refs or not isinstance(clause_refs, list):
        clause_refs = [format_clause_ref(c) for c in clauses_used]
    else:
        clause_refs = [format_clause_ref(c) for c in clause_refs]
    return {
        "decision": llm_output.get("decision", "pending"),
        "covered_amount": llm_output.get("covered_amount", "Unknown"),
        "patient_responsibility": llm_output.get("patient_responsibility", "Unknown"),
        "justification": llm_output.get("justification", "No justification provided."),
        "clause_refs": clause_refs,
        "confidence": llm_output.get("confidence", 0.5)
    }

def mistral_decision(query, demographics, duration_days, clauses):
    prompt_prefix = f'''You are an expert insurance policy analyst. Given the following user query and relevant policy clauses, return a JSON with:
- decision: (approved/denied/pending)
- covered_amount: (e.g., "$8,000")
- patient_responsibility: (e.g., "$4,000")
- justification: (explain the decision)
- clause_refs: (list of objects, each with document, section, and text fields, referencing the clauses used for the decision)
- confidence: (a number between 0 and 1 indicating your confidence in the decision)

User Query: "{query}"
Demographics: {json.dumps(demographics)}
Policy Duration (days): {duration_days}
Clauses:
{json.dumps(clauses, indent=2)}

Return JSON:
'''
    # Truncate clauses to fit context window
    truncated_clauses = truncate_clauses_to_fit(clauses, prompt_prefix, max_tokens=2048, buffer=200)
    prompt = prompt_prefix + json.dumps([c for c in truncated_clauses], indent=2) + "\n\nReturn JSON:\n"
    try:
        output = llm(prompt, max_tokens=512)
        text = output['choices'][0]['text'] if isinstance(output, dict) and 'choices' in output else str(output)
        if text.strip().startswith("```"):
            lines = text.strip().splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        parsed = json.loads(text)
        parsed['model_used'] = 'mistral'
    except Exception as e:
        parsed = {"decision": "error", "covered_amount": None, "patient_responsibility": None, "justification": f"Mistral error: {str(e)}. Raw response: {text if 'text' in locals() else ''}", "clause_refs": [], "confidence": 0.0, "model_used": "mistral"}
    return parsed

def cohere_decision(query, demographics, duration_days, clauses):
    prompt = f'''
You are an expert insurance policy analyst. Given the following user query and relevant policy clauses, return a JSON with:
- decision: (approved/denied/pending)
- covered_amount: (e.g., "$8,000")
- patient_responsibility: (e.g., "$4,000")
- justification: (explain the decision)
- clause_refs: (list of objects, each with document, section, and text fields, referencing the clauses used for the decision)
- confidence: (a number between 0 and 1 indicating your confidence in the decision)

User Query: "{query}"
Demographics: {json.dumps(demographics)}
Policy Duration (days): {duration_days}
Clauses:
{json.dumps(clauses, indent=2)}

Return JSON:
'''
    try:
        response = co.generate(
            model='command-r-plus',  # or another Cohere model if desired
            prompt=prompt,
            max_tokens=512,
            temperature=0.2
        )
        text = response.generations[0].text.strip()
        if text.startswith("```"):
            text = text.strip('`').strip()
        parsed = json.loads(text)
        parsed['model_used'] = 'cohere'
    except Exception as e:
        parsed = {"decision": "error", "justification": f"Cohere error: {str(e)}", "clause_refs": [], "confidence": 0.0, "model_used": "cohere"}
    return parsed

def gemini_decision(query, demographics, duration_days, clauses):
    import json
    prompt = f'''
You are an expert insurance policy analyst. Given the following user query and relevant policy clauses, return a JSON with:
- decision: (approved/denied/pending)
- covered_amount: (e.g., "$8,000")
- patient_responsibility: (e.g., "$4,000")
- justification: (explain the decision)
- clause_refs: (list of objects, each with document, section, and text fields, referencing the clauses used for the decision)
- confidence: (a number between 0 and 1 indicating your confidence in the decision)

User Query: "{query}"
Demographics: {json.dumps(demographics)}
Policy Duration (days): {duration_days}
Clauses:
{json.dumps(clauses, indent=2)}

Return JSON:
'''
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 512, "temperature": 0.2}
    }
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        text = result["candidates"][0]["content"]["parts"][0]["text"] if "candidates" in result and result["candidates"] else ""
        print("Gemini raw response:", text)  # Logging for debugging
        # Strip Markdown code block if present
        if text.strip().startswith("```"):
            lines = text.strip().splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
            parsed['model_used'] = 'gemini'
        except Exception as e:
            print("Gemini JSON parse error, falling back to Cohere:", e)
            parsed = cohere_decision(query, demographics, duration_days, clauses)
            if parsed.get("decision") == "error":
                print("Cohere error, falling back to Mistral:", parsed.get("justification"))
                parsed = mistral_decision(query, demographics, duration_days, clauses)
    except requests.exceptions.Timeout:
        print("Gemini API request timed out, falling back to Cohere.")
        parsed = cohere_decision(query, demographics, duration_days, clauses)
        if parsed.get("decision") == "error":
            print("Cohere error, falling back to Mistral:", parsed.get("justification"))
            parsed = mistral_decision(query, demographics, duration_days, clauses)
    except Exception as e:
        print("Gemini error, falling back to Cohere:", e)
        parsed = cohere_decision(query, demographics, duration_days, clauses)
        if parsed.get("decision") == "error":
            print("Cohere error, falling back to Mistral:", parsed.get("justification"))
            parsed = mistral_decision(query, demographics, duration_days, clauses)
    return parsed

@app.route('/batch_query', methods=['POST'])
def handle_batch_query():
    data = request.get_json(force=True)
    questions = data.get('questions', [])
    responses = []
    for question in questions:
        cache_key = question.strip().lower()
        cached = cache_get(cache_key)
        if cached:
            responses.append(redact_pii_in_dict(cached))
            audit_log({"timestamp": datetime.now(timezone.utc).isoformat(), "endpoint": "/batch_query", "question": question, "response": redact_pii_in_dict(cached)})
            continue
        # Synchronous call to existing handle_query logic
        with app.test_request_context('/query', method='POST', json={"question": question}):
            resp = handle_query()
            result = resp.get_json()
            cache_set(cache_key, result)
            redacted_result = redact_pii_in_dict(result)
            audit_log({"timestamp": datetime.now(timezone.utc).isoformat(), "endpoint": "/batch_query", "question": question, "response": redacted_result})
            responses.append(redacted_result)
    return jsonify({"results": responses})

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    question = data.get('question', '').strip()
    cache_key = question.lower() + '_concise'
    cached = cache_get(cache_key)
    if cached:
        response = redact_pii_in_dict(cached)
        audit_log({"timestamp": datetime.now(timezone.utc).isoformat(), "endpoint": "/query", "question": question, "response": response})
        return jsonify(response)
    expander = QueryExpander(llm=llm)
    try:
        expansion_result = expander.expand(question)
        expanded_queries = expansion_result.get('expansions', [question])
    except Exception as e:
        expanded_queries = [question]
    section_match = None
    section_pattern = re.search(r'section\s*(\d+(?:\.\d+)*)', question, re.IGNORECASE)
    if section_pattern:
        section_match = section_pattern.group(1)
    all_results = []
    for q in expanded_queries:
        results = collection.query(query_texts=[q], n_results=10)
        if results['documents'] and results['documents'][0]:
            all_results.append(results)
            break
    if not all_results:
        concise_answer = "No relevant information found."
        cache_set(cache_key, {"answer": concise_answer})
        redacted_response = redact_pii_in_dict({"answer": concise_answer})
        audit_log({"timestamp": datetime.now(timezone.utc).isoformat(), "endpoint": "/query", "question": question, "response": redacted_response})
        return jsonify(redacted_response)
    results = all_results[0]
    context_chunks = results['documents'][0] if results['documents'] and results['documents'][0] else []
    metadatas = results['metadatas'][0] if results['metadatas'] and results['metadatas'][0] else []
    scored_chunks = []
    for chunk, meta in zip(context_chunks, metadatas):
        score = 0.0
        if section_match:
            if (meta.get('section_headers') and section_match in meta.get('section_headers')) or (section_match in chunk):
                score += 1.0
        scored_chunks.append((chunk, meta, score))
    scored_chunks.sort(key=lambda x: x[2], reverse=True)
    top_chunks = scored_chunks[:3]
    clauses_used = []
    for chunk, meta, score in top_chunks:
        clauses_used.append({
            "document": meta.get("source", ""),
            "section": (meta.get("section_headers", "") or ""),
            "text": chunk,
            "relevance_score": score,
            "effective_dates": meta.get("effective_dates", ""),
            "footnotes": meta.get("footnotes", ""),
            "tables": meta.get("tables", "")
        })
    concise_prompt = f'You are an expert insurance policy analyst. Given the following user query and relevant policy clauses, answer concisely in one sentence, e.g., "Yes, knee surgery is covered under the policy." or "No, this is not covered." Do not provide justification or details unless necessary for clarity. The answer must be a single line.\nUser Query: "{question}"\nClauses:\n{json.dumps(clauses_used, indent=2)}\n'
    concise_answer = None
    # Try Gemini first
    try:
        concise_answer = gemini_generate(concise_prompt, max_tokens=48, temperature=0.1)
        if (not concise_answer or 'error' in concise_answer.lower() or 'timed out' in concise_answer.lower()):
            raise Exception('Gemini failed')
    except Exception:
        # Fallback to Cohere
        try:
            response = co.generate(
                model='command-r-plus',
                prompt=concise_prompt,
                max_tokens=48,
                temperature=0.1
            )
            concise_answer = response.generations[0].text.strip()
            if not concise_answer:
                raise Exception('Cohere failed')
        except Exception:
            concise_answer = "No relevant information found."
    concise_answer = concise_answer.strip().replace('\n', ' ')
    # Ensure only one line
    concise_answer = concise_answer.split('\n')[0].strip()
    cache_set(cache_key, {"answer": concise_answer})
    redacted_response = redact_pii_in_dict({"answer": concise_answer})
    audit_log({"timestamp": datetime.now(timezone.utc).isoformat(), "endpoint": "/query", "question": question, "response": redacted_response})
    return jsonify(redacted_response)

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    data = request.get_json()
    feedback_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": data.get("question"),
        "system_response": data.get("system_response"),
        "user_feedback": data.get("user_feedback")
    }
    with open("feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback_entry) + "\n")
    return jsonify({"status": "success", "message": "Feedback recorded."})

if __name__ == '__main__':
    # Simple server run without debug
    # If you see Windows console errors, try running with: python -u backend/app.py
    app.run(host='0.0.0.0', port=5001, debug=False)