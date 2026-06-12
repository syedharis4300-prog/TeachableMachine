import os
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import uuid

# Import ModelManager
try:
    from model_manager import ModelManager
except ImportError:
    from backend.model_manager import ModelManager

# Setup base directory for data storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

app = FastAPI(
    title="Neural Studio Backend",
    description="Machine Learning service for custom image classification",
    version="1.0.0"
)

# Enable CORS for frontend interface communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate Model Manager
model_manager = ModelManager(data_dir=DATA_DIR)

class TrainRequest(BaseModel):
    epochs: int = 50
    lr: float = 0.001
    batch_size: int = 16

@app.get("/health")
def health_check():
    """Simple API healthcheck."""
    return {
        "status": "healthy",
        "device": str(model_manager.device),
        "model_loaded": model_manager.classifier is not None,
        "classes_configured": len(model_manager.classes)
    }

@app.get("/classes")
def get_classes():
    """Retrieve all configured training classes and their current image counts."""
    return {
        "classes": model_manager.classes,
        "sample_counts": model_manager.get_sample_counts()
    }

@app.post("/classes")
def add_class(class_name: str = Query(..., description="Name of the class to add")):
    """Add a new classification class."""
    try:
        model_manager.add_class(class_name)
        return {
            "status": "success", 
            "classes": model_manager.classes,
            "sample_counts": model_manager.get_sample_counts()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/classes/{class_name}")
def delete_class(class_name: str):
    """Delete a classification class and all its associated training images."""
    if class_name not in model_manager.classes:
        raise HTTPException(status_code=404, detail=f"Class '{class_name}' not found.")
    
    model_manager.remove_class(class_name)
    return {
        "status": "success", 
        "classes": model_manager.classes,
        "sample_counts": model_manager.get_sample_counts()
    }

@app.post("/upload/{class_name}")
async def upload_samples(class_name: str, files: List[UploadFile] = File(...)):
    """Upload one or more training images to the specified class."""
    if class_name not in model_manager.classes:
        raise HTTPException(status_code=404, detail=f"Class '{class_name}' not configured.")
    
    success_count = 0
    errors = []
    
    for file in files:
        # Validate file type
        content_type = file.content_type or ""
        if not content_type.startswith("image/"):
            errors.append(f"Skipped file '{file.filename}': not an image.")
            continue
            
        try:
            contents = await file.read()
            # Construct a unique safe filename
            unique_filename = f"{uuid.uuid4()}_{file.filename}"
            model_manager.save_image_sample(class_name, contents, unique_filename)
            success_count += 1
        except Exception as e:
            errors.append(f"Failed processing file '{file.filename}': {str(e)}")
            
    if errors and success_count == 0:
        raise HTTPException(status_code=400, detail="; ".join(errors))
        
    return {
        "status": "success",
        "uploaded_count": success_count,
        "failed_count": len(errors),
        "errors": errors,
        "sample_counts": model_manager.get_sample_counts()
    }

@app.post("/train")
def train_model(req: TrainRequest):
    """Trigger background training of the classification head."""
    try:
        model_manager.start_training_job(
            epochs=req.epochs,
            lr=req.lr,
            batch_size=req.batch_size
        )
        return {
            "status": "started",
            "message": "Model training loop has started in the background."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/train/status")
def get_training_status():
    """Retrieve the current state, progress and loss/accuracy metrics of the training job."""
    return model_manager.get_training_status()

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Perform real-time inference on an uploaded image."""
    if not model_manager.classifier:
        raise HTTPException(
            status_code=400, 
            detail="Inference model is not trained. Add samples and train first."
        )
        
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
        
    try:
        contents = await file.read()
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        predictions = model_manager.predict(image)
        # Find the class with maximum confidence
        top_class = max(predictions, key=predictions.get)
        top_confidence = predictions[top_class]
        
        return {
            "status": "success",
            "predictions": predictions,
            "top_class": top_class,
            "top_confidence": top_confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction inference failed: {str(e)}")

@app.get("/export")
def export_model():
    """Export the trained PyTorch state dict and classes metadata as a ZIP archive."""
    try:
        zip_buffer = model_manager.export_model_zip()
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=teachable_model.zip"}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/import")
async def import_model(file: UploadFile = File(...)):
    """Import a previously exported model ZIP archive (containing classes.json and classifier.pt)."""
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip archives are allowed.")
        
    try:
        contents = await file.read()
        model_manager.import_model_zip(contents)
        return {
            "status": "success",
            "message": "Model and metadata imported successfully.",
            "classes": model_manager.classes,
            "sample_counts": model_manager.get_sample_counts()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/reset")
def reset_state():
    """Reset the backend state by deleting all classes, training images, and trained classifiers."""
    try:
        model_manager.reset_state()
        return {
            "status": "success",
            "message": "Backend session state wiped successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
