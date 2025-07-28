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
from io import BytesIO
import tempfile
from PyPDF2 import PdfReader
import docx

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
        decision = "pendingvi"
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
        
        print(f"\n=== PROCESSING QUESTION: {question} ===")
        
        # IMPROVED EVIDENCE RETRIEVAL - Much more aggressive and comprehensive
        # Get many more candidates for better coverage
        results = collection.query(query_texts=[question], n_results=30)
        
        if not results['documents'] or not results['documents'][0]:
            return jsonify({"answer": "No relevant information found in the policy documents."})
        
        # Extract ALL meaningful keywords from question
        question_lower = question.lower()
        question_words = [word.strip('.,?!()[]{}"') for word in question_lower.split() if len(word.strip('.,?!()[]{}"')) > 2]
        
        # Enhanced keyword extraction based on question type
        all_keywords = set(question_words)
        if 'grace' in question_lower or 'period' in question_lower:
            all_keywords.update(['grace', 'period', 'grace period', 'thirty', '30', 'days', 'payment', 'premium', 'due', 'renewal', 'continue', 'continuity'])
        if 'premium' in question_lower:
            all_keywords.update(['premium', 'payment', 'due', 'grace', 'renewal', 'continue', 'continuity'])
        if 'parent' in question_lower or 'dependent' in question_lower:
            all_keywords.update(['parent', 'parents', 'dependent', 'dependents', 'family', 'spouse', 'children'])
        if 'waiting' in question_lower:
            all_keywords.update(['waiting', 'period', 'exclusion', 'months', 'days'])
        if 'coverage' in question_lower or 'covered' in question_lower:
            all_keywords.update(['coverage', 'covered', 'benefit', 'include', 'exclude'])
        
        # Score and rank chunks with multiple strategies
        scored_chunks = []
        for i, (chunk, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            score = 0.0
            chunk_lower = chunk.lower()
            
            # Keyword matching score (most important)
            keyword_matches = sum(1 for keyword in all_keywords if keyword in chunk_lower)
            score += keyword_matches * 3.0
            
            # Exact phrase matching (very important)
            if 'grace period' in question_lower and 'grace period' in chunk_lower:
                score += 10.0
            if 'thirty days' in chunk_lower or '30 days' in chunk_lower:
                score += 8.0
            if 'premium payment' in question_lower and 'premium' in chunk_lower and 'payment' in chunk_lower:
                score += 8.0
            
            # Section relevance
            if metadata and metadata.get('section_headers'):
                score += 2.0
            
            # Length bonus for substantial chunks
            if len(chunk) > 200:
                score += 1.0
            
            scored_chunks.append({
                'text': chunk,
                'metadata': metadata or {},
                'score': score,
                'keyword_matches': keyword_matches
            })
        
        # Sort by score and take top chunks
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)
        top_chunks = scored_chunks[:15]
        
        print(f"Top 5 chunks by score:")
        for i, chunk in enumerate(top_chunks[:5]):
            print(f"Chunk {i+1} (score: {chunk['score']}, keywords: {chunk['keyword_matches']}): {chunk['text'][:200]}...")
        
        # Prepare evidence with more context (800 chars instead of 400)
        evidence_chunks = []
        total_tokens = 0
        for chunk in top_chunks:
            chunk_text = chunk['text'][:800] + ("..." if len(chunk['text']) > 800 else "")
            tokens = len(chunk_text) // 4
            if total_tokens + tokens > 600:  # Increased token limit
                break
            evidence_chunks.append({
                'text': chunk_text,
                'section': chunk['metadata'].get('section_headers', 'Policy Document'),
                'score': chunk['score']
            })
            total_tokens += tokens
        
        # IMPROVED PROMPT - Much more specific and balanced
        enhanced_prompt = f'''You are an expert insurance policy analyst. Based on the policy clauses below, answer the user's question accurately and comprehensively.

IMPORTANT INSTRUCTIONS:
- If the information exists in the clauses, provide a detailed answer starting with "Yes," or "No," as appropriate
- Include specific details like amounts, time periods, conditions, and section references
- If coverage exists, explain the conditions and limits clearly
- If something is excluded, explain why and reference the exclusion
- Maximum 2 lines, but be comprehensive and specific
- Only say "The policy does not specify" if the information is truly not present

User Question: "{question}"

Policy Clauses:
{json.dumps(evidence_chunks, indent=2)}

Detailed Answer:'''
        
        # Generate answer with fallback logic
        final_answer = None
        model_used = None
        
        # Try Gemini first with higher token limit
        try:
            final_answer = gemini_generate(enhanced_prompt, max_tokens=150, temperature=0.1)
            if final_answer and 'error' not in final_answer.lower() and 'timed out' not in final_answer.lower():
                model_used = "gemini"
            else:
                raise Exception('Gemini failed or returned error')
        except Exception as e:
            print(f"Gemini error: {str(e)}")
            # Fallback to Cohere
            try:
                response = co.generate(
                    model='command-r-plus',
                    prompt=enhanced_prompt,
                    max_tokens=150,
                    temperature=0.1
                )
                final_answer = response.generations[0].text.strip()
                if final_answer:
                    model_used = "cohere"
                else:
                    raise Exception('Cohere returned empty response')
            except Exception as e:
                print(f"Cohere error: {str(e)}")
                final_answer = "Unable to process the question due to technical issues."
                model_used = "none"
        
        # Clean up answer
        if final_answer:
            final_answer = final_answer.strip()
            # Ensure max 2 lines
            lines = final_answer.split('\n')
            if len(lines) > 2:
                final_answer = '\n'.join(lines[:2])
        
        print(f"Final answer: {final_answer}")
        print(f"Model used: {model_used}")
        print("=== END PROCESSING ===")
        
        # Build response
        response_data = {
            "answer": final_answer,
            "model_used": model_used,
            "response_time_ms": tracker.get_response_time_ms()
        }
        
        cache_set(cache_key, response_data)
        audit_log({"timestamp": datetime.now(timezone.utc).isoformat(), "endpoint": "/query", "question": question, "response": response_data})
        
        return jsonify({"answer": final_answer})
        
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

@app.route('/hackrx/run', methods=['POST'])
def hackrx_run():
    """
    Accepts a JSON body with:
    {
        "question": "...",
        "documents": ["url1", "url2", ...]
    }
    Downloads and processes the documents at runtime, then answers the question using the same logic as /query.
    """
    tracker = PerformanceTracker()
    tracker.start()
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        question = data.get('question', '').strip()
        doc_urls = data.get('documents', [])
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        # Handle both string and array formats for documents
        if not doc_urls:
            return jsonify({"error": "No documents provided"}), 400
        
        # Convert string to list if needed
        if isinstance(doc_urls, str):
            doc_urls = [doc_urls]
        elif not isinstance(doc_urls, list):
            return jsonify({"error": "Documents should be a string URL or list of URLs"}), 400

        # Download and extract text from each document
        all_texts = []
        for url in doc_urls:
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                content_type = resp.headers.get('Content-Type', '').lower()
                if '.pdf' in url.lower() or 'pdf' in content_type:
                    with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
                        tmp.write(resp.content)
                        tmp.flush()
                        reader = PdfReader(tmp.name)
                        text = "\n".join(page.extract_text() or '' for page in reader.pages)
                        all_texts.append(text)
                elif '.docx' in url.lower() or 'word' in content_type or 'docx' in content_type:
                    with tempfile.NamedTemporaryFile(suffix='.docx') as tmp:
                        tmp.write(resp.content)
                        tmp.flush()
                        doc = docx.Document(tmp.name)
                        text = "\n".join([p.text for p in doc.paragraphs])
                        all_texts.append(text)
                else:
                    # Fallback: treat as plain text
                    all_texts.append(resp.text)
            except Exception as e:
                print(f"Error downloading or processing {url}: {e}")
                continue
        if not all_texts:
            return jsonify({"error": "No valid documents could be processed."}), 400

        # Combine all texts for chunking
        combined_text = "\n\n".join(all_texts)
        # Use the improved section/paragraph-based chunking
        from utils.query_expander import QueryExpander  # for llm if needed
        processor = ClauseRetriever(None)  # We'll use a local embedding function
        # Use the same chunking logic as ingest.py
        def semantic_chunking(text):
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 0]
            chunks = []
            for para in paragraphs:
                if len(para.split()) < MIN_CHUNK_LENGTH:
                    continue
                if len(para.split()) > CHUNK_SIZE:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    current_chunk = []
                    current_len = 0
                    for sent in sentences:
                        sent_len = len(sent.split())
                        if current_len + sent_len > CHUNK_SIZE and current_chunk:
                            chunk_text = ' '.join(current_chunk)
                            chunks.append({'text': chunk_text})
                            current_chunk = []
                            current_len = 0
                        current_chunk.append(sent)
                        current_len += sent_len
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append({'text': chunk_text})
                else:
                    chunks.append({'text': para})
            return chunks

        # Chunk the combined text
        chunks = semantic_chunking(combined_text)
        if not chunks:
            return jsonify({"error": "No valid text chunks found in documents."}), 400

        # Embed the chunks using the same embedding model as before
        from chromadb.utils import embedding_functions
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=str(EMBEDDING_MODEL)
        )
        chunk_texts = [c['text'] for c in chunks]
        chunk_embeddings = embedding_fn(chunk_texts)

        # Embed the question
        question_embedding = embedding_fn([question])[0]

        # Compute cosine similarity and get top N chunks
        import numpy as np
        def cosine_sim(a, b):
            a = np.array(a)
            b = np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        scored_chunks = []
        for i, emb in enumerate(chunk_embeddings):
            score = cosine_sim(question_embedding, emb)
            scored_chunks.append({'text': chunk_texts[i], 'relevance_score': score})
        # Sort and take top 20 for filtering
        top_chunks = sorted(scored_chunks, key=lambda x: x['relevance_score'], reverse=True)[:20]

        # Enhanced aggressive keyword filtering to capture ALL relevant terms
        # Extract ALL meaningful words from the question
        question_lower = question.lower()
        question_words = [word.strip('.,?!()[]{}"') for word in question_lower.split() if len(word.strip('.,?!()[]{}"')) > 2]
        
        # Specific term extraction based on question content
        question_keywords = []
        if 'parent' in question_lower:
            question_keywords.extend(['parent', 'parents', 'dependent', 'dependents', 'family'])
        if 'child' in question_lower:
            question_keywords.extend(['child', 'children', 'dependent', 'dependents'])
        if 'spouse' in question_lower:
            question_keywords.extend(['spouse', 'husband', 'wife', 'dependent', 'dependents'])
        if 'grace' in question_lower or 'period' in question_lower:
            question_keywords.extend(['grace', 'period', 'grace period', 'thirty', '30', 'days', 'payment', 'premium', 'due', 'renewal'])
        if 'premium' in question_lower:
            question_keywords.extend(['premium', 'payment', 'due', 'grace', 'renewal', 'continue', 'continuity'])
        if 'waiting' in question_lower:
            question_keywords.extend(['waiting', 'period', 'exclusion', 'months', 'days'])
        if 'coverage' in question_lower or 'covered' in question_lower:
            question_keywords.extend(['coverage', 'covered', 'benefit', 'include', 'exclude'])
        
        # General insurance keywords (comprehensive list)
        general_keywords = [
            "cover", "covered", "not covered", "included", "excluded", "benefit", "payable", 
            "eligible", "allowed", "not allowed", "waiting period", "limit", "sum insured", 
            "definition", "means", "shall mean", "grace", "period", "premium", "payment", 
            "due", "renewal", "continue", "continuity", "thirty", "days", "months", "years"
        ]
        
        # Combine ALL keywords: question words + specific terms + general terms
        all_keywords = list(set(question_words + question_keywords + general_keywords))
        print(f"\n=== KEYWORD MATCHING DEBUG ===")
        print(f"Question: {question}")
        print(f"Question keywords: {question_keywords}")
        print(f"All keywords: {all_keywords}")
        
        keyword_chunks = []
        for c in top_chunks:
            matched_keywords = [kw for kw in all_keywords if kw in c['text'].lower()]
            if matched_keywords:
                c['matched_keywords'] = matched_keywords
                keyword_chunks.append(c)
                print(f"Chunk matched keywords: {matched_keywords}")
                print(f"Chunk text: {c['text'][:150]}...")
        
        print(f"Found {len(keyword_chunks)} keyword-rich chunks out of {len(top_chunks)} total")
        print("=== END KEYWORD DEBUG ===")
        
        # Improved fallback logic - always ensure we have good chunks
        if keyword_chunks and len(keyword_chunks) >= 5:
            # Use keyword-rich chunks if we have enough
            filtered_chunks = keyword_chunks[:12]
            print(f"Using {len(filtered_chunks)} keyword-rich chunks")
        elif keyword_chunks:
            # Mix keyword chunks with top chunks for better coverage
            filtered_chunks = keyword_chunks + [c for c in top_chunks if c not in keyword_chunks]
            filtered_chunks = filtered_chunks[:12]
            print(f"Using mixed chunks: {len(keyword_chunks)} keyword + {len(filtered_chunks) - len(keyword_chunks)} top chunks")
        else:
            # Fallback to top chunks but take more for better coverage
            filtered_chunks = top_chunks[:15]  # Increased from 8 to 15
            print(f"Using fallback: {len(filtered_chunks)} top chunks without keyword filtering")

        # Optimize chunks for better context while maintaining latency (increased from 400 to 800 chars)
        optimized_clauses = []
        current_tokens = 0
        for clause in filtered_chunks:
            # Preserve more context - 800 chars instead of 400
            short_text = clause['text'][:800] + ("..." if len(clause['text']) > 800 else "")
            tokens = len(short_text) // 4
            if current_tokens + tokens > 400:  # Increased token limit
                break
            # Try to preserve section metadata if available
            section_info = 'Unknown'
            if hasattr(clause, 'metadata') and clause.metadata:
                section_info = clause.metadata.get('section_headers', 'Unknown')
            elif 'metadata' in clause:
                section_info = clause['metadata'].get('section_headers', 'Unknown')
            
            optimized_clauses.append({
                'text': short_text, 
                'metadata': {'section_headers': section_info}
            })
            current_tokens += tokens

        # Print evidence chunks for debugging
        print("---- Evidence Chunks ----")
        for i, e in enumerate(optimized_clauses):
            print(f"Chunk {i+1}: {e['text'][:300]}")
            print(f"Section: {e['metadata'].get('section_headers', 'Unknown')}")
            print("------------------------")

        # Add debugging to see what evidence is being provided
        print(f"\n=== DEBUGGING QUESTION: {question} ===")
        print(f"Number of evidence chunks: {len(optimized_clauses)}")
        for i, clause in enumerate(optimized_clauses):
            print(f"Evidence {i+1}: {clause['text'][:200]}...")
        print("=== END DEBUGGING ===")
        
        # Use the improved balanced prompt that doesn't bias towards "No"
        concise_prompt = f'''You are an expert insurance policy analyst. Analyze the policy clauses carefully and answer the user's question accurately.

IMPORTANT INSTRUCTIONS:
- Carefully read ALL policy clauses provided before answering
- Look for ANY mention of coverage, benefits, or definitions related to the question
- Start with "Yes," if coverage exists OR "No," if explicitly excluded
- Include specific conditions, limits, waiting periods, and requirements when coverage exists
- Always mention the relevant policy section/clause reference (e.g., "Section C, Part A.9")
- Be definitive and direct - avoid phrases like "it appears", "seems to", "may be", "possibly"
- If coverage exists: "Yes, [item] is covered [conditions/limits], [requirements]. (Section reference)"
- If explicitly excluded: "No, [item] is excluded [reason]. (Section reference)"
- Maximum 2 lines, be comprehensive but concise
- Only use "Coverage is not mentioned in the policy" if absolutely no relevant information exists
- IMPORTANT: Look for coverage in definitions, benefits, dependents sections - not just exclusions

User Query: "{question}"

Policy Clauses:
{json.dumps([{"text": c['text'], "section": c['metadata'].get('section_headers', 'Unknown')} for c in optimized_clauses], indent=2)}

Detailed Answer:'''

        concise_answer = None
        model_used = None
        try:
            concise_answer = gemini_generate(concise_prompt, max_tokens=120, temperature=0.1)
            if (not concise_answer or 'error' in concise_answer.lower() or 'timed out' in concise_answer.lower()):
                raise Exception('Gemini failed')
            model_used = "gemini"
        except Exception as e:
            print(f"Gemini error: {str(e)}")
            try:
                response = co.generate(
                    model='command-r-plus',
                    prompt=concise_prompt,
                    max_tokens=120,
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

        # Clean answer but preserve multi-line format for descriptive answers
        concise_answer = concise_answer.strip()
        
        # Only do minimal post-processing if the answer doesn't start correctly
        if concise_answer and not concise_answer.startswith(('Yes,', 'No,', 'Coverage is not')):
            # Check if it's a clear positive or negative answer and only then modify
            answer_lower = concise_answer.lower()
            if answer_lower.startswith('yes') and not concise_answer.startswith('Yes,'):
                concise_answer = f"Yes, {concise_answer[3:].lstrip(',').strip()}"
            elif answer_lower.startswith('no') and not concise_answer.startswith('No,'):
                concise_answer = f"No, {concise_answer[2:].lstrip(',').strip()}"
            elif 'not mentioned' in answer_lower or 'no information' in answer_lower:
                concise_answer = "Coverage is not mentioned in the policy."
            # Remove the aggressive defaulting that was causing incorrect "No" answers
        
        # Ensure answer is concise (max 2 lines)
        lines = concise_answer.split('\n')
        if len(lines) > 2:
            concise_answer = '\n'.join(lines[:2])
        # Also limit to reasonable word count (max 50 words for 2 lines)
        words = concise_answer.split()
        if len(words) > 50:
            concise_answer = ' '.join(words[:50])

        response_data = {
            "answer": concise_answer,
            "model_used": model_used,
            "response_time_ms": tracker.get_response_time_ms()
        }
        return jsonify(response_data)
    except Exception as e:
        import traceback
        error_msg = f"Unexpected error in /hackrx/run: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({"error": error_msg, "response_time_ms": tracker.get_response_time_ms()}), 500

if __name__ == '__main__':
    # Simple server run without debug
    # If you see Windows console errors, try running with: python -u backend/app.py
    app.run(host='0.0.0.0', port=5001, debug=False)