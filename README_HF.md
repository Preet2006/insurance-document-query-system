# Insurance Document Query System - Hugging Face Spaces

This is an LLM-Powered Intelligent Query–Retrieval System for processing insurance documents and answering natural language queries.

## 🚀 Live Demo

The system is deployed on Hugging Face Spaces and can be accessed at:
`https://huggingface.co/spaces/[YOUR_USERNAME]/insurance-document-query-system`

## 📋 API Endpoints

### Health Check
```
GET /health
```

### Hackathon Endpoint
```
POST /hackrx/run
Content-Type: application/json
```

#### Request Format:
```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "Does this policy cover maternity expenses?"
  ]
}
```

#### Response Format:
```json
{
  "answers": [
    "A grace period of thirty days is provided...",
    "Yes, the policy covers maternity expenses..."
  ]
}
```

## 🔧 Technical Stack

- **Backend**: Flask
- **Vector Database**: ChromaDB (in-memory)
- **Embeddings**: Sentence Transformers (BAAI/bge-small-en-v1.5)
- **LLM**: Gemini API (with Cohere/Mistral fallbacks)
- **Document Processing**: PDF, DOCX, EML support
- **Deployment**: Hugging Face Spaces (Docker)

## 📁 Project Structure

```
├── app.py                 # Hugging Face Spaces entry point
├── requirements.txt       # Python dependencies
├── Dockerfile.hf         # Docker configuration for HF Spaces
├── backend/
│   ├── app.py           # Main Flask application
│   ├── config.py        # Configuration settings
│   └── utils/           # Utility modules
└── data/                # Sample documents
```

## 🚀 Deployment

This application is deployed on Hugging Face Spaces using Docker. The deployment automatically:

1. Installs system dependencies (Tesseract OCR, Poppler)
2. Installs Python dependencies
3. Sets up the Flask application
4. Exposes the API on port 7860

## 🔍 Features

- **Multi-format Document Support**: PDF, DOCX, EML files
- **Semantic Search**: Advanced embedding-based retrieval
- **Intelligent Chunking**: Paragraph/section-based document processing
- **Multi-LLM Support**: Gemini, Cohere, and Mistral APIs
- **Memory Optimized**: Efficient processing for cloud deployment
- **Structured Output**: JSON responses with explainable decisions

## 📊 Performance

- **Accuracy**: High precision query understanding and clause matching
- **Latency**: Optimized for real-time performance
- **Token Efficiency**: Optimized LLM token usage
- **Memory Usage**: Optimized for cloud deployment constraints

## 🔐 Environment Variables

The following environment variables can be set in Hugging Face Spaces:

- `GEMINI_API_KEY`: Your Gemini API key
- `COHERE_API_KEY`: Your Cohere API key (optional, for fallback)
- `MISTRAL_API_KEY`: Your Mistral API key (optional, for fallback)

## 🧪 Testing

You can test the API using curl or any HTTP client:

```bash
# Health check
curl https://your-space-url.hf.space/health

# Test the hackathon endpoint
curl -X POST https://your-space-url.hf.space/hackrx/run \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": ["What is the grace period for premium payment?"]
  }'
```

## 📝 License

MIT License - see LICENSE file for details. 