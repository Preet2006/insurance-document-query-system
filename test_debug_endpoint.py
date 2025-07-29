import requests
import json

def test_debug_endpoint():
    """Debug test to see full error response"""
    
    test_data = {
        "documents": "https://httpbin.org/robots.txt",
        "questions": [
            "What is the grace period for premium payment?"
        ]
    }
    
    try:
        response = requests.post(
            "http://localhost:5001/hackrx/run",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Full Response Text: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Error Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_debug_endpoint() 