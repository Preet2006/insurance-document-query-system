import requests
import json

def test_mock_policy():
    """Test with mock insurance policy data"""
    
    # Create a mock insurance policy document
    mock_policy = """
    NATIONAL PARIVAR MEDICLAIM PLUS POLICY
    
    SECTION 1: GRACE PERIOD
    A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.
    
    SECTION 2: PRE-EXISTING DISEASES
    There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.
    
    SECTION 3: MATERNITY COVERAGE
    Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.
    
    SECTION 4: CATARACT SURGERY
    The policy has a specific waiting period of two (2) years for cataract surgery.
    
    SECTION 5: ORGAN DONOR COVERAGE
    Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.
    
    SECTION 6: NO CLAIM DISCOUNT
    A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.
    
    SECTION 7: PREVENTIVE HEALTH CHECK
    Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.
    
    SECTION 8: HOSPITAL DEFINITION
    A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.
    
    SECTION 9: AYUSH TREATMENTS
    The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.
    
    SECTION 10: ROOM RENT AND ICU LIMITS
    For Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN).
    """
    
    # Save mock policy to a temporary file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(mock_policy)
        temp_file = f.name
    
    try:
        # Create a simple HTTP server to serve the file
        import http.server
        import socketserver
        import threading
        import time
        
        # Start a simple HTTP server in a thread
        def start_server():
            with socketserver.TCPServer(("", 8000), http.server.SimpleHTTPRequestHandler) as httpd:
                httpd.serve_forever()
        
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        time.sleep(1)  # Wait for server to start
        
        # Test the endpoint
        test_data = {
            "documents": "http://localhost:8000/" + os.path.basename(temp_file),
            "questions": [
                "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
                "What is the waiting period for pre-existing diseases (PED) to be covered?",
                "Does this policy cover maternity expenses, and what are the conditions?"
            ]
        }
        
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
            
            if "answers" in result and isinstance(result["answers"], list):
                print(f"✅ Got {len(result['answers'])} answers:")
                for i, answer in enumerate(result["answers"]):
                    print(f"Answer {i+1}: {answer}")
        else:
            print(f"❌ Error Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    finally:
        # Clean up
        try:
            os.unlink(temp_file)
        except:
            pass

if __name__ == "__main__":
    test_mock_policy() 