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
import time

app = Flask(__name__)
CORS(app)

# === COHERE API KEY SETUP ===
# Paste your Cohere API key below (or load from environment/config securely)
COHERE_API_KEY = "qZmghdKw7d7YxNryMj57OsMN0jLsQSCy0c7xulRA"
co = cohere.Client(COHERE_API_KEY)

def llm(prompt, max_tokens=512, temperature=0.2):
    response = co.generate(
        model='command-r-plus',
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.generations[0].text

# === PERFORMANCE TRACKING ===
class PerformanceTracker:
    def __init__(self):
        self.start_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def get_response_time_ms(self):
        if self.start_time:
            return int((time.time() - self.start_time) * 1000)
        return 0

# === ACCURACY IMPROVEMENTS ===
class ClauseRetriever:
    def __init__(self, collection):
        self.collection = collection
    
    def get_relevant_clauses(self, query, n_results=5):
        """Enhanced clause retrieval with better scoring"""
        try:
            # Get more candidates first
            results = self.collection.query(query_texts=[query], n_results=15)
            
            if not results['documents'] or not results['documents'][0]:
                return []
            
            # Re-rank using multiple strategies
            scored_chunks = []
            query_words = set(query.lower().split())
            
            for chunk, meta in zip(results['documents'][0], results['metadatas'][0]):
                score = 0.0
                chunk_lower = chunk.lower()
                
                # Exact keyword matching
                for word in query_words:
                    if word in chunk_lower:
                        score += 2.0
                
                # Section relevance
                if meta.get('section_headers'):
                    score += 1.0
                
                # Document source relevance
                if meta.get('source'):
                    score += 0.5
                
                scored_chunks.append({
                    'text': chunk,
                    'metadata': meta,
                    'relevance_score': score
                })
            
            # Return top N most relevant
            return sorted(scored_chunks, key=lambda x: x['relevance_score'], reverse=True)[:n_results]
            
        except Exception as e:
            print(f"Clause retrieval error: {str(e)}")
            return []

class ResponseFormatter:
    @staticmethod
    def format_clause_evidence(clause_data):
        """Format clause data for JSON response"""
        return {
            "clause": clause_data['metadata'].get('section_headers', 'Unknown Section'),
            "text": clause_data['text'][:300] + "..." if len(clause_data['text']) > 300 else clause_data['text'],
            "relevance_score": round(clause_data['relevance_score'], 2),
            "document": clause_data['metadata'].get('source', 'Unknown Document')
        }
    
    @staticmethod
    def calculate_accuracy_score(clauses_used, answer):
        """Calculate accuracy score based on evidence quality"""
        if not clauses_used:
            return 0.0
        
        # Base score from relevance scores
        avg_relevance = sum(c['relevance_score'] for c in clauses_used) / len(clauses_used)
        
        # Boost score if answer is confident
        confidence_boost = 0.2 if any(word in answer.lower() for word in ['yes', 'covered', 'include']) else 0.0
        
        return min(1.0, avg_relevance / 5.0 + confidence_boost)

# === TOKEN EFFICIENCY ===
class TokenOptimizer:
    @staticmethod
    def truncate_clauses_aggressively(clauses, max_tokens=1000):
        """Only send the most relevant parts of clauses"""
        truncated = []
        current_tokens = 0
        
        for clause in clauses:
            # Take only first 200 characters of each clause
            short_text = clause['text'][:200] + "..."
            tokens = len(short_text) // 4
            
            if current_tokens + tokens > max_tokens:
                break
                
            truncated.append({
                **clause,
                'text': short_text
            })
            current_tokens += tokens
        
        return truncated

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

# Initialize our new components
clause_retriever = ClauseRetriever(collection)
response_formatter = ResponseFormatter()
token_optimizer = TokenOptimizer()

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
    # Initialize performance tracker
    tracker = PerformanceTracker()
    tracker.start()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        question = data.get('question', '').strip()
        if not question:
            return jsonify({"error": "No question provided"}), 400
            
        cache_key = question.lower() + '_enhanced'
        cached = cache_get(cache_key)
        if cached:
            cached['response_time_ms'] = tracker.get_response_time_ms()
            response = redact_pii_in_dict(cached)
            audit_log({"timestamp": datetime.now(timezone.utc).isoformat(), "endpoint": "/query", "question": question, "response": response})
            return jsonify(response)
        
        # Initialize QueryExpander with error handling
        try:
            expander = QueryExpander(llm=llm)
            expansion_result = expander.expand(question)
            expanded_queries = expansion_result.get('expansions', [question])
        except Exception as e:
            print(f"QueryExpander error: {str(e)}")
            expanded_queries = [question]
        
        # Enhanced clause retrieval
        clauses_data = clause_retriever.get_relevant_clauses(question, n_results=8)
        
        if not clauses_data:
            concise_answer = "No relevant information found."
            accuracy_score = 0.0
            evidence = []
        else:
            # Optimize tokens for LLM
            # Aggressively truncate each clause to 400 chars for latency
            optimized_clauses = []
            current_tokens = 0
            for clause in clauses_data:
                short_text = clause['text'][:400] + ("..." if len(clause['text']) > 400 else "")
                tokens = len(short_text) // 4
                if current_tokens + tokens > 250:
                    break
                optimized_clauses.append({**clause, 'text': short_text})
                current_tokens += tokens
            
            # Generate answer with enhanced prompt
            concise_prompt = f'''You are an expert insurance policy analyst. Given the following user query and relevant policy clauses, answer in a single line starting with 'Yes,' or 'No,' as appropriate, followed by a brief reason or key detail (such as coverage amount, waiting period, or main condition). If the information is not present, reply: 'Coverage for [topic] is not specified in the policy.' Do not use hedging language. Be concise and specific.

User Query: "{question}"

Clauses:
{json.dumps([{"text": c['text'], "section": c['metadata'].get('section_headers', 'Unknown')} for c in optimized_clauses], indent=2)}

Answer:'''
            
            concise_answer = None
            model_used = None
            
            # Try Gemini first
            try:
                concise_answer = gemini_generate(concise_prompt, max_tokens=48, temperature=0.1)
                if (not concise_answer or 'error' in concise_answer.lower() or 'timed out' in concise_answer.lower()):
                    raise Exception('Gemini failed')
                model_used = "gemini"
            except Exception as e:
                print(f"Gemini error: {str(e)}")
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
                    model_used = "cohere"
                except Exception as e:
                    print(f"Cohere error: {str(e)}")
                    concise_answer = "No relevant information found."
                    model_used = "none"
            
            # Calculate accuracy score
            accuracy_score = response_formatter.calculate_accuracy_score(clauses_data, concise_answer)
            
            # Format evidence
            evidence = [response_formatter.format_clause_evidence(c) for c in clauses_data]
            print("---- Evidence Chunks ----")
            for i, e in enumerate(evidence):
                print(f"Chunk {i+1}: {e['text'][:300]}")
                print(f"Section: {e['clause']}, Document: {e['document']}")
                print("------------------------")
        
        # Clean up answer
        concise_answer = concise_answer.strip().replace('\n', ' ')
        concise_answer = concise_answer.split('\n')[0].strip()
        
        # Build enhanced response (keep full data for cache and audit)
        response_data = {
            "answer": concise_answer,
            "accuracy_score": round(accuracy_score, 2),
            "evidence": evidence,
            "model_used": model_used,
            "response_time_ms": tracker.get_response_time_ms()
        }
        
        cache_set(cache_key, response_data)
        redacted_response = redact_pii_in_dict(response_data)
        audit_log({"timestamp": datetime.now(timezone.utc).isoformat(), "endpoint": "/query", "question": question, "response": redacted_response})
        
        # Return only the answer field to the user
        return jsonify({"answer": redacted_response["answer"]})
        
    except Exception as e:
        import traceback
        error_msg = f"Unexpected error in handle_query: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({"error": error_msg, "response_time_ms": tracker.get_response_time_ms()}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    error_msg = f"Server error: {str(e)}\n{traceback.format_exc()}"
    print(error_msg)
    return jsonify({"error": error_msg}), 500

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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cache_size": len(query_cache)
    })

if __name__ == '__main__':
    # Simple server run without debug
    # If you see Windows console errors, try running with: python -u backend/app.py
    app.run(host='0.0.0.0', port=5001, debug=False)