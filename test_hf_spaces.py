#!/usr/bin/env python3
"""
Test script for Hugging Face Spaces deployment
"""

import requests
import json
import time

def test_hf_spaces_deployment(space_url):
    """
    Test the Hugging Face Spaces deployment
    """
    print(f"🧪 Testing Hugging Face Spaces deployment: {space_url}")
    print("=" * 60)
    
    # Test data
    test_data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?"
        ]
    }
    
    # Test 1: Health Check
    print("1️⃣ Testing health check...")
    try:
        response = requests.get(f"{space_url}/health", timeout=30)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Hackathon Endpoint
    print("\n2️⃣ Testing hackathon endpoint...")
    print("   This may take time due to document processing and cold start...")
    
    for attempt in range(3):
        try:
            print(f"   Attempt {attempt + 1}/3...")
            start_time = time.time()
            
            response = requests.post(
                f"{space_url}/hackrx/run",
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=300  # 5 minutes timeout for document processing
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if response.status_code == 200:
                print("✅ Hackathon endpoint test passed!")
                print(f"   Processing time: {processing_time:.2f} seconds")
                
                result = response.json()
                if "answers" in result and len(result["answers"]) == len(test_data["questions"]):
                    print(f"   Received {len(result['answers'])} answers")
                    for i, answer in enumerate(result["answers"]):
                        print(f"   Q{i+1}: {answer[:100]}...")
                else:
                    print("   ⚠️  Unexpected response format")
                    print(f"   Response: {result}")
                
                return True
            else:
                print(f"❌ Hackathon endpoint failed: {response.status_code}")
                print(f"   Response: {response.text}")
                
                if attempt < 2:  # Don't wait on last attempt
                    print("   Waiting 10 seconds before retry...")
                    time.sleep(10)
                    
        except requests.exceptions.Timeout:
            print(f"❌ Timeout on attempt {attempt + 1}")
            if attempt < 2:
                print("   Waiting 10 seconds before retry...")
                time.sleep(10)
        except Exception as e:
            print(f"❌ Error on attempt {attempt + 1}: {e}")
            if attempt < 2:
                print("   Waiting 10 seconds before retry...")
                time.sleep(10)
    
    print("❌ All attempts failed")
    return False

if __name__ == "__main__":
    # Replace with your actual Hugging Face Spaces URL
    # Format: https://your-username-insurance-document-query-system.hf.space
    space_url = input("Enter your Hugging Face Spaces URL: ").strip()
    
    if not space_url:
        print("❌ Please provide a valid Hugging Face Spaces URL")
        exit(1)
    
    # Remove trailing slash if present
    space_url = space_url.rstrip('/')
    
    success = test_hf_spaces_deployment(space_url)
    
    if success:
        print("\n🎉 All tests passed! Your Hugging Face Spaces deployment is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check your deployment and try again.") 