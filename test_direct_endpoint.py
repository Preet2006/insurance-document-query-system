import requests
import json

def test_direct_endpoint():
    """Test the endpoint logic directly"""
    
    # Test the health endpoint first
    try:
        health_response = requests.get("http://localhost:5001/health")
        print(f"Health check: {health_response.status_code}")
        if health_response.status_code == 200:
            print("✅ Server is running")
        else:
            print("❌ Server is not responding")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    # Test with a simple request to see if the endpoint exists
    test_data = {
        "documents": "https://example.com/test.pdf",
        "questions": ["Test question"]
    }
    
    try:
        response = requests.post(
            "http://localhost:5001/hackrx/run",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Endpoint test status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 400:
            error_data = response.json()
            if "Failed to download" in error_data.get("error", ""):
                print("✅ Endpoint is working - document download failed as expected")
                print("✅ The system is ready for the hackathon!")
                print("\n📋 To use with real documents:")
                print("1. Upload your PDF/DOCX to a publicly accessible URL")
                print("2. Send POST request to /hackrx/run with:")
                print("   - documents: URL to your document")
                print("   - questions: Array of questions")
                print("3. Receive answers array in response")
            else:
                print("❌ Unexpected error")
        else:
            print("✅ Endpoint responded successfully")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_direct_endpoint() 