# 🚀 Deploy to Render - Complete Guide

## Prerequisites
- GitHub account
- Render account (free at [render.com](https://render.com))
- API keys for Gemini and Cohere (optional, system has fallbacks)

## Step 1: Push Code to GitHub

1. **Initialize Git** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit for deployment"
   ```

2. **Create GitHub Repository**:
   - Go to [github.com](https://github.com)
   - Click "New repository"
   - Name it `insurance-document-query-system`
   - Make it public or private
   - Don't initialize with README (we already have one)

3. **Push to GitHub**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/insurance-document-query-system.git
   git branch -M main
   git push -u origin main
   ```

## Step 2: Deploy on Render

### Option A: Using Render Dashboard (Recommended)

1. **Sign up/Login to Render**:
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the `insurance-document-query-system` repository

3. **Configure the Service**:
   - **Name**: `insurance-document-query-system`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `cd backend && python app.py`
   - **Plan**: Free (or paid for better performance)

4. **Set Environment Variables** (optional):
   - Click "Environment" tab
   - Add these variables:
     - `GEMINI_API_KEY`: Your Gemini API key (optional - system has fallbacks)
     - `PYTHON_VERSION`: `3.10.0`
   - **Note**: Cohere API key is already configured in the app

5. **Deploy**:
   - Click "Create Web Service"
   - Wait for build to complete (5-10 minutes)

### Option B: Using render.yaml (Advanced)

1. **The render.yaml file is already created** in your project
2. **Push to GitHub** (if you haven't already)
3. **On Render Dashboard**:
   - Click "New +" → "Blueprint"
   - Connect your repository
   - Render will automatically detect and use the render.yaml

## Step 3: Get Your Deployment URL

After deployment, Render will give you a URL like:
```
https://insurance-document-query-system.onrender.com
```

**Save this URL** - you'll need it for the hackathon submission!

## Step 4: Test Your Deployment

1. **Test Health Endpoint**:
   ```
   GET https://your-app-name.onrender.com/health
   ```
   Should return: `{"status": "healthy", ...}`

2. **Test Hackathon Endpoint**:
   ```bash
   python test_deployment.py
   ```
   (Update the URL in the script first)

3. **Manual Test**:
   ```bash
   curl -X POST https://your-app-name.onrender.com/hackrx/run \
     -H "Content-Type: application/json" \
     -d '{
       "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
       "questions": ["What is the grace period for premium payment?"]
     }'
   ```

## Step 5: Hackathon Submission

For the hackathon, you'll need to provide:

1. **Your deployed URL**: `https://your-app-name.onrender.com`
2. **Endpoint**: `/hackrx/run`
3. **Method**: `POST`
4. **Headers**: `Content-Type: application/json`
5. **Body Format**:
   ```json
   {
     "documents": "https://example.com/policy.pdf",
     "questions": ["Question 1", "Question 2"]
   }
   ```

## Troubleshooting

### Common Issues:

1. **Build Fails**:
   - Check that all dependencies are in `backend/requirements.txt`
   - Ensure Python version is compatible

2. **App Won't Start**:
   - Check logs in Render dashboard
   - Verify start command is correct
   - Ensure port is set correctly (Render auto-sets PORT env var)

3. **API Keys Not Working**:
   - Cohere API key is already configured in the app
   - Only set GEMINI_API_KEY if you want to use Gemini (optional)
   - System has fallbacks, so it should still work without any API keys

4. **Cold Start Issues**:
   - Free tier has cold starts (first request takes 30-60 seconds)
   - Consider upgrading to paid plan for better performance

### Performance Tips:

1. **Upgrade to Paid Plan**: Better performance, no cold starts
2. **Use CDN**: Consider Cloudflare for global edge caching
3. **Optimize Dependencies**: Remove unused packages from requirements.txt

## Monitoring

- **Logs**: Available in Render dashboard
- **Metrics**: Response times, errors, etc.
- **Uptime**: Render provides uptime monitoring

## Security Notes

- ✅ API keys are stored as environment variables (secure)
- ✅ No sensitive data in code
- ✅ HTTPS is automatically enabled
- ✅ CORS is configured for API access

## Cost

- **Free Tier**: $0/month (with limitations)
- **Paid Plans**: Start at $7/month for better performance

---

## 🎉 You're Ready!

Your system is now deployed and ready for the hackathon evaluation. The endpoint will handle:
- ✅ PDF, DOCX, EML documents
- ✅ Single or multiple questions
- ✅ Runtime document processing
- ✅ Structured JSON responses
- ✅ High accuracy and low latency

**Good luck with the hackathon! 🚀** 