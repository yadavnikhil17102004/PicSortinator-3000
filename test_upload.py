#!/usr/bin/env python3
"""
🧪 Test upload functionality for PicSortinator 3000
"""

import requests
import os
import time

def test_connection():
    """Test if Flask app is running"""
    try:
        response = requests.get('http://localhost:5000')
        print(f"📊 GET / status: {response.status_code}")
        return True
    except Exception as e:
        print(f"💥 Connection error: {e}")
        return False

def test_upload():
    """Test uploading a file to the Flask app"""
    # First check connection
    if not test_connection():
        print("❌ Flask app not accessible!")
        return
    
    url = 'http://localhost:5000/upload'
    
    # Check if test image exists
    test_image = 'test_image.jpg'
    if not os.path.exists(test_image):
        print("❌ Test image not found! Creating one...")
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        img.save(test_image)
        print("✅ Test image created!")
    
    try:
        # Test GET request to upload page first
        response = requests.get(url)
        print(f"📊 GET /upload status: {response.status_code}")
        
        # Now try POST request
        with open(test_image, 'rb') as f:
            files = {'files[]': (test_image, f, 'image/jpeg')}
            response = requests.post(url, files=files)
            
        print(f"📊 POST /upload status: {response.status_code}")
        print(f"📄 Response content: {response.text[:500]}...")
        
        if response.status_code == 200:
            print("✅ Upload test successful!")
        else:
            print("❌ Upload test failed!")
            
    except Exception as e:
        print(f"💥 Error during upload test: {e}")

if __name__ == "__main__":
    print("🧪 Testing PicSortinator 3000 upload functionality...")
    test_upload()
