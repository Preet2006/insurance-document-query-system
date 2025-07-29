# 🚀 Hugging Face Spaces Deployment Guide

This guide will help you deploy your Insurance Document Query System to Hugging Face Spaces, which offers better resources than Render's free tier for ML workloads.

## 📋 Prerequisites

1. **Hugging Face Account**: Create one at https://huggingface.co/join
2. **Git Repository**: Your code should be in a Git repository (GitHub, GitLab, etc.)
3. **API Keys**: Ensure you have your Gemini API key ready

## 🎯 Step-by-Step Deployment

### Step 1: Create a New Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in the details:
   - **Owner**: Your username
   - **Space name**: `insurance-document-query-system` (or any name you prefer)
   - **License**: MIT
   - **Space SDK**: **Docker** ⭐ (Important!)
   - **Space hardware**: **CPU** (free tier)
   - **Visibility**: Public (recommended for hackathon)

### Step 2: Connect Your Repository

1. In the Space creation page, choose **"Repository"** as the source
2. Select your Git repository (GitHub/GitLab)
3. Choose the branch (usually `main` or `master`)
4. Click **"Create Space"**

### Step 3: Configure Environment Variables

1. Go to your Space's **Settings** tab
2. Scroll down to **"Repository secrets"**
3. Add the following secrets:
   - `GEMINI_API_KEY`: Your Gemini API key
   - `COHERE_API_KEY`: Your Cohere API key (optional, for fallback)
   - `MISTRAL_API_KEY`: Your Mistral API key (optional, for fallback)

### Step 4: Wait for Build

1. Hugging Face will automatically start building your Space
2. The build process will:
   - Install system dependencies (Tesseract OCR, Poppler)
   - Install Python dependencies from `requirements.txt`
   - Set up the Docker container
   - Start the Flask application

3. Monitor the build logs in the **"Logs"** tab
4. Build time: Usually 5-10 minutes for the first build

### Step 5: Verify Deployment

1. Once the build is complete, your Space will be available at:
   ```
   https://your-username-insurance-document-query-system.hf.space
   ```

2. Test the health endpoint:
   ```bash
   curl https://your-username-insurance-document-query-system.hf.space/health
   ```

3. Run the test script:
   ```bash
   python test_hf_spaces.py
   ```

## 🔧 File Structure for Hugging Face Spaces

Your repository should have these files in the root directory:

```
├── app.py                 # Entry point for HF Spaces
├── requirements.txt       # Python dependencies
├── Dockerfile.hf         # Docker configuration
├── README_HF.md          # Documentation
├── test_hf_spaces.py     # Test script
├── backend/              # Your existing backend code
│   ├── app.py
│   ├── config.py
│   └── utils/
└── data/                 # Sample documents
```

## 🚨 Troubleshooting

### Build Failures

1. **System Dependencies**: Ensure `Dockerfile.hf` includes all necessary system packages
2. **Python Dependencies**: Check `requirements.txt` for version conflicts
3. **Memory Issues**: Hugging Face Spaces have more RAM than Render free tier

### Runtime Issues

1. **Port Configuration**: The app uses port 7860 (HF Spaces default)
2. **Environment Variables**: Ensure all API keys are set in Space secrets
3. **Cold Start**: First request may take longer due to model loading

### Common Errors

1. **Import Errors**: Check that all dependencies are in `requirements.txt`
2. **Permission Errors**: Ensure proper file permissions in Docker
3. **Memory Errors**: Optimize model loading and chunk processing

## 📊 Performance Optimization

### Memory Optimization (Already Implemented)

- ✅ Singleton embedding function
- ✅ Limited chunk processing (max 20 chunks)
- ✅ Reduced evidence text length (300 chars)
- ✅ Optimized PDF chunking

### Additional Optimizations for HF Spaces

1. **Model Caching**: Models are cached between requests
2. **Lazy Loading**: Embedding model loads only when needed
3. **Request Timeouts**: 5-minute timeout for document processing

## 🧪 Testing Your Deployment

### Quick Test

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

### Full Test Suite

Run the provided test script:
```bash
python test_hf_spaces.py
```

## 🔄 Updating Your Deployment

1. **Push Changes**: Commit and push changes to your Git repository
2. **Auto-Redeploy**: Hugging Face Spaces automatically redeploys on new commits
3. **Monitor Logs**: Check the **"Logs"** tab for build and runtime logs

## 📈 Advantages of Hugging Face Spaces

- ✅ **More RAM**: Better than Render's 512MB limit
- ✅ **ML-Optimized**: Designed for machine learning workloads
- ✅ **Free GPU**: Available for some use cases
- ✅ **Easy Deployment**: Simple Git integration
- ✅ **Community**: Great for hackathons and demos
- ✅ **No Credit Card**: Completely free

## 🎉 Success!

Once deployed, your API will be available at:
```
https://your-username-insurance-document-query-system.hf.space
```

**API Endpoints:**
- `GET /health` - Health check
- `POST /hackrx/run` - Main hackathon endpoint

Your system is now ready for the hackathon evaluation! 🚀 