import requests
import json

def test_hackrx_endpoint():
    """Test the /hackrx/run endpoint with sample data"""
    
    # Test data matching the hackathon format
    test_data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?"
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
    test_hackrx_endpoint() 