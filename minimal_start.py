#!/usr/bin/env python3
"""
Minimal startup script for the Flask app
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def main():
    """Minimal startup function"""
    print("🚀 Starting minimal Flask app...")
    
    # Get port from environment variable (for production) or use 5001
    port = int(os.environ.get('PORT', 5001))
    print(f"Port: {port}")
    
    try:
        # Import the Flask app
        from app import app
        print("✅ Flask app imported successfully")
        
        # Start the server
        print(f"Starting server on port {port}...")
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 