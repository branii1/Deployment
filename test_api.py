import requests
import json

# Test the running API
def test_api():
    base_url = "http://localhost:8080"
    
    try:
        # Test home endpoint
        print("🏠 Testing home endpoint...")
        response = requests.get(f"{base_url}/")
        print(f"Home: {response.status_code} - {response.json()}")
        
        # Test health endpoint
        print("❤️ Testing health endpoint...")
        response = requests.get(f"{base_url}/health")
        print(f"Health: {response.status_code} - {response.json()}")
        
        # Test prediction endpoint
        print("🎯 Testing prediction endpoint...")
        test_data = {
            "feature1": 2020,
            "feature2": 90, 
            "feature3": 5,
            "feature4": 0,
            "feature5": 0,
            "feature6": 0
        }
        
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Predict: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
    except Exception as e:
        print(f"❌ Error testing API: {e}")

if __name__ == "__main__":
    test_api()