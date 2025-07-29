import requests
import json
import os

def test_deployed_endpoint():
    """Test the deployed endpoint on Render"""
    
    # Get the deployed URL from environment or use a placeholder
    deployed_url = os.getenv('DEPLOYED_URL', 'https://insurance-document-query-system.onrender.com')
    
    print(f"Testing deployed endpoint: {deployed_url}")
    
    # Test health endpoint first
    # Try health check with retries for cold start
    print("Testing health endpoint (may take 30-60 seconds on first request due to cold start)...")
    for attempt in range(3):
        try:
            health_response = requests.get(f"{deployed_url}/health", timeout=30)
            print(f"Health check status: {health_response.status_code}")
            if health_response.status_code == 200:
                print("✅ Health check passed")
                break
            elif health_response.status_code == 502:
                print(f"⚠️  Cold start in progress (attempt {attempt + 1}/3), waiting...")
                import time
                time.sleep(20)  # Wait 20 seconds for cold start
                continue
            else:
                print(f"❌ Health check failed with status {health_response.status_code}")
                return
        except Exception as e:
            print(f"⚠️  Connection attempt {attempt + 1}/3 failed: {e}")
            if attempt < 2:
                print("Waiting 10 seconds before retry...")
                import time
                time.sleep(10)
            else:
                print("❌ Cannot connect to deployed server after 3 attempts")
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
    
    # Test the hackathon endpoint with retries
    print("Testing hackathon endpoint (may take time due to document processing)...")
    for attempt in range(10):  # More retries for cold start
        try:
            print(f"Attempt {attempt + 1}/10...")
            response = requests.post(
                f"{deployed_url}/hackrx/run",
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=600  # Even longer timeout for document processing
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
                break  # Success, exit retry loop
            else:
                print(f"❌ Error Status: {response.status_code}")
                print(f"❌ Error Response: {response.text}")
                if attempt < 1:  # If not the last attempt
                    print("Retrying...")
                    import time
                    time.sleep(10)
                else:
                    print("❌ Failed after all attempts")
                    
        except Exception as e:
            print(f"❌ Test failed: {e}")
            if attempt < 1:
                print("Retrying...")
                import time
                time.sleep(10)

if __name__ == "__main__":
    test_deployed_endpoint() 