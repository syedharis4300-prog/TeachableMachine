import os
import io
import time
import shutil
from PIL import Image
from fastapi.testclient import TestClient

# Adjust path to find main
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import app, model_manager

client = TestClient(app)

def create_dummy_image(color):
    """Generate a simple colored image for testing."""
    img = Image.new("RGB", (224, 224), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()

def test_full_flow():
    print("Starting integration test for Teachable Machine Backend...")
    
    # 1. Reset the state to start clean
    print("\n--- Step 1: Reset state ---")
    response = client.post("/reset")
    assert response.status_code == 200
    print("Reset response:", response.json())
    
    # 2. Add classes
    print("\n--- Step 2: Add training classes ---")
    response = client.post("/classes?class_name=RedClass")
    assert response.status_code == 200
    
    response = client.post("/classes?class_name=BlueClass")
    assert response.status_code == 200
    
    response = client.get("/classes")
    assert response.status_code == 200
    classes_info = response.json()
    print("Configured classes:", classes_info)
    assert "RedClass" in classes_info["classes"]
    assert "BlueClass" in classes_info["classes"]
    
    # 3. Upload image samples
    print("\n--- Step 3: Uploading training samples ---")
    # Generate 5 red images and 5 blue images
    for i in range(5):
        red_img = create_dummy_image("red")
        response = client.post(
            "/upload/RedClass",
            files={"files": (f"red_{i}.jpg", red_img, "image/jpeg")}
        )
        assert response.status_code == 200
        
        blue_img = create_dummy_image("blue")
        response = client.post(
            "/upload/BlueClass",
            files={"files": (f"blue_{i}.jpg", blue_img, "image/jpeg")}
        )
        assert response.status_code == 200
        
    response = client.get("/classes")
    classes_info = response.json()
    print("Class sample counts after upload:", classes_info["sample_counts"])
    assert classes_info["sample_counts"]["RedClass"] == 5
    assert classes_info["sample_counts"]["BlueClass"] == 5
    
    # 4. Trigger training
    print("\n--- Step 4: Trigger model training ---")
    response = client.post("/train", json={"epochs": 15, "lr": 0.01, "batch_size": 4})
    assert response.status_code == 200
    print("Train response:", response.json())
    
    # Poll training status until complete
    print("Polling training status...")
    status = "training"
    for _ in range(30): # Timeout after 30 seconds
        time.sleep(1)
        response = client.get("/train/status")
        assert response.status_code == 200
        data = response.json()
        status = data["status"]
        progress = data["progress"]
        print(f"Status: {status} | Progress: {progress:.2f} | Epoch: {data['current_epoch']}")
        if status in ["completed", "failed"]:
            break
            
    assert status == "completed", f"Training failed or timed out. Last state: {data}"
    print("Training metrics:", data["metrics"])
    
    # 5. Test inference (predictions)
    print("\n--- Step 5: Test predictions ---")
    # Test a red image
    test_red = create_dummy_image("red")
    response = client.post(
        "/predict",
        files={"file": ("test_red.jpg", test_red, "image/jpeg")}
    )
    assert response.status_code == 200
    pred_data = response.json()
    print("Red image prediction:", pred_data)
    assert pred_data["top_class"] == "RedClass"
    assert pred_data["predictions"]["RedClass"] > 0.7
    
    # Test a blue image
    test_blue = create_dummy_image("blue")
    response = client.post(
        "/predict",
        files={"file": ("test_blue.jpg", test_blue, "image/jpeg")}
    )
    assert response.status_code == 200
    pred_data = response.json()
    print("Blue image prediction:", pred_data)
    assert pred_data["top_class"] == "BlueClass"
    assert pred_data["predictions"]["BlueClass"] > 0.7
    
    # 6. Export model ZIP
    print("\n--- Step 6: Export model ---")
    response = client.get("/export")
    assert response.status_code == 200
    zip_bytes = response.content
    print(f"Exported ZIP file size: {len(zip_bytes)} bytes")
    assert len(zip_bytes) > 0
    
    # 7. Import model ZIP (verify persistence)
    print("\n--- Step 7: Import model and verify ---")
    # Reset first
    client.post("/reset")
    
    # Import ZIP
    response = client.post(
        "/import",
        files={"file": ("model.zip", zip_bytes, "application/zip")}
    )
    assert response.status_code == 200
    import_data = response.json()
    print("Import response:", import_data)
    assert "RedClass" in import_data["classes"]
    
    # Verify predictions still work after importing
    test_red = create_dummy_image("red")
    response = client.post(
        "/predict",
        files={"file": ("test_red.jpg", test_red, "image/jpeg")}
    )
    assert response.status_code == 200
    pred_data = response.json()
    print("Red image prediction after import:", pred_data)
    assert pred_data["top_class"] == "RedClass"
    
    print("\nAll integration tests passed successfully!")

if __name__ == "__main__":
    test_full_flow()
