#!/usr/bin/env python3
"""
Hugging Face Spaces Entry Point
This file serves as the entry point for Hugging Face Spaces deployment.
It imports and runs the Flask app from the backend module.
"""

import os
import sys

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Import the Flask app from backend
from app import app

if __name__ == "__main__":
    # Get port from environment variable (Hugging Face Spaces sets this)
    port = int(os.environ.get("PORT", 7860))
    
    # Run the Flask app
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    ) 