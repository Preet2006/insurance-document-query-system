#!/usr/bin/env python3
"""
Basic Flask test app
"""

import os
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return {'message': 'Hello from Flask!'}

@app.route('/health')
def health():
    return {'status': 'ok'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting basic Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False) 