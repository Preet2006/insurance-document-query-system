import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Model Config
EMBEDDING_MODEL = BASE_DIR / "models/bge-small-en-v1.5"
LLM_MODEL = BASE_DIR / "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Gemini API Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent")

# ChromaDB Config
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "insurance_policies_enhanced"

# Processing Config
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
MIN_CHUNK_LENGTH = 50

# Retrieval Config
BASE_RESULTS = 5
EXPANDED_RESULTS = 3
SIMILARITY_THRESHOLD = 0.3
