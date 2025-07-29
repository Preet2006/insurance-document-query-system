import requests
import json

def test_simple_endpoint():
    """Test the /hackrx/run endpoint with simple mock data"""
    
    # Test data with a simple text document
    test_data = {
        "documents": "https://httpbin.org/robots.txt",  # Simple text endpoint for testing
        "questions": [
            "What is the grace period for premium payment?",
            "Does this policy cover maternity expenses?",
            "What is the waiting period for pre-existing diseases?"
        ]
    }
    
    # Make request to the endpoint
    try:
        response = requests.post(
            "http://localhost:5001/hackrx/run",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Validate response format
            if "answers" in result and isinstance(result["answers"], list):
                print(f"✅ Success! Got {len(result['answers'])} answers")
                for i, answer in enumerate(result["answers"]):
                    print(f"Answer {i+1}: {answer}")
            else:
                print("❌ Error: Response doesn't contain 'answers' array")
        else:
            print(f"❌ Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")

if __name__ == "__main__":
    test_simple_endpoint() 