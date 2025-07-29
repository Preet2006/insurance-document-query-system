import requests
import json
import os

def test_deployed_endpoint():
    """Test the deployed endpoint on Render"""
    
    # Get the deployed URL from environment or use a placeholder
    deployed_url = os.getenv('DEPLOYED_URL', 'https://your-app-name.onrender.com')
    
    print(f"Testing deployed endpoint: {deployed_url}")
    
    # Test health endpoint first
    try:
        health_response = requests.get(f"{deployed_url}/health", timeout=10)
        print(f"Health check status: {health_response.status_code}")
        if health_response.status_code == 200:
            print("✅ Health check passed")
        else:
            print("❌ Health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to deployed server: {e}")
        return
    
    # Test the hackathon endpoint
    test_data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?"
        ]
    }
    
    try:
        response = requests.post(
            f"{deployed_url}/hackrx/run",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        print(f"Endpoint test status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Response: {json.dumps(result, indent=2)}")
            
            if "answers" in result and isinstance(result["answers"], list):
                print(f"✅ Got {len(result['answers'])} answers:")
                for i, answer in enumerate(result["answers"]):
                    print(f"Answer {i+1}: {answer}")
            else:
                print("❌ Response doesn't contain 'answers' array")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_deployed_endpoint() 