import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import numpy as np
import threading
import shutil
import io

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # Simple classification head: Linear layer followed by optional dropout
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim, num_classes)
        )
        
    def forward(self, x):
        return self.fc(x)

class ModelManager:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.classes_dir = os.path.join(data_dir, "classes")
        self.model_path = os.path.join(data_dir, "classifier.pt")
        self.classes_file = os.path.join(data_dir, "classes.json")
        
        os.makedirs(self.classes_dir, exist_ok=True)
        
        # Load pre-trained feature extractor (MobileNetV3 Small)
        # Use CPU by default, since local execution will run on user's machine
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load weights safely using weights argument
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self.feature_extractor = models.mobilenet_v3_small(weights=weights)
        except Exception:
            # Fallback for older torchvision versions
            self.feature_extractor = models.mobilenet_v3_small(pretrained=True)
            
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Freeze all feature extractor layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # Image transformation pipeline (ImageNet standards)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Classifier state
        self.classifier = None
        self.classes = []  # List of class names (indices correspond to classifier output indices)
        self.load_classes_metadata()
        self.load_classifier_weights()
        
        # Training state and lock
        self.training_status = {
            "status": "idle",
            "progress": 0.0,
            "current_epoch": 0,
            "total_epochs": 0,
            "metrics": {
                "loss": [],
                "accuracy": []
            },
            "error": None
        }
        self.training_lock = threading.Lock()
        
    def load_classes_metadata(self):
        """Load class names and map from JSON if exists."""
        if os.path.exists(self.classes_file):
            try:
                with open(self.classes_file, "r") as f:
                    self.classes = json.load(f)
            except Exception as e:
                print(f"Error loading classes file: {e}")
                self.classes = []
        else:
            self.classes = []
            
    def save_classes_metadata(self):
        """Save class names to JSON."""
        with open(self.classes_file, "w") as f:
            json.dump(self.classes, f, indent=4)
            
    def load_classifier_weights(self):
        """Load trained classifier weights if they exist."""
        if os.path.exists(self.model_path) and self.classes:
            try:
                # Feature dimension for MobileNetV3 Small features
                feat_dim = 576
                num_classes = len(self.classes)
                self.classifier = Classifier(feat_dim, num_classes)
                self.classifier.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.classifier.to(self.device)
                self.classifier.eval()
                print("Trained model weights loaded successfully.")
            except Exception as e:
                print(f"Error loading classifier weights: {e}")
                self.classifier = None
        else:
            self.classifier = None
            
    def add_class(self, class_name):
        """Add a new class and create its folder."""
        clean_name = class_name.strip()
        if not clean_name:
            raise ValueError("Class name cannot be empty")
        if clean_name in self.classes:
            return  # Already exists
            
        self.classes.append(clean_name)
        os.makedirs(os.path.join(self.classes_dir, clean_name), exist_ok=True)
        self.save_classes_metadata()
        # Invalidate current classifier as class list changed
        self.classifier = None
        
    def remove_class(self, class_name):
        """Remove a class and its folder."""
        if class_name in self.classes:
            self.classes.remove(class_name)
            class_path = os.path.join(self.classes_dir, class_name)
            if os.path.exists(class_path):
                shutil.rmtree(class_path)
            self.save_classes_metadata()
            # Invalidate current classifier
            self.classifier = None
            if os.path.exists(self.model_path):
                try:
                    os.remove(self.model_path)
                except Exception:
                    pass
                    
    def save_image_sample(self, class_name, image_bytes, filename):
        """Save an image sample under the class's directory."""
        if class_name not in self.classes:
            self.add_class(class_name)
            
        class_path = os.path.join(self.classes_dir, class_name)
        image_path = os.path.join(class_path, filename)
        
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image.save(image_path, "JPEG")
            return True
        except Exception as e:
            raise ValueError(f"Failed to process and save image: {e}")
            
    def get_sample_counts(self):
        """Get the count of image samples per class."""
        counts = {}
        for c in self.classes:
            class_path = os.path.join(self.classes_dir, c)
            if os.path.exists(class_path):
                files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
                counts[c] = len(files)
            else:
                counts[c] = 0
        return counts

    def extract_features(self, image: Image.Image) -> torch.Tensor:
        """Extract a 576-dimensional feature vector for a PIL Image using MobileNetV3."""
        # Preprocess the image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Pass through feature extractor conv layers
            features = self.feature_extractor.features(img_tensor)
            # Global Average Pooling
            features = self.feature_extractor.avgpool(features)
            # Flatten to 1D vector (dim: 576)
            features = torch.flatten(features, 1)
            
        return features.squeeze(0).cpu()

    def start_training_job(self, epochs=50, lr=0.001, batch_size=16):
        """Start model training in a background thread."""
        with self.training_lock:
            if self.training_status["status"] == "training":
                raise RuntimeError("Training is already in progress.")
                
            self.training_status = {
                "status": "training",
                "progress": 0.0,
                "current_epoch": 0,
                "total_epochs": epochs,
                "metrics": {
                    "loss": [],
                    "accuracy": []
                },
                "error": None
            }
            
        # Spawn training thread
        thread = threading.Thread(
            target=self._run_training,
            args=(epochs, lr, batch_size)
        )
        thread.start()
        
    def _run_training(self, epochs, lr, batch_size):
        """Background thread execution of model training."""
        try:
            # 1. Load all images and extract their features
            dataset_features = []
            dataset_labels = []
            
            sample_counts = self.get_sample_counts()
            
            # Validation checks
            if len(self.classes) < 2:
                raise ValueError("Must have at least 2 classes configured to train a classifier.")
            
            for class_idx, class_name in enumerate(self.classes):
                if sample_counts.get(class_name, 0) == 0:
                    raise ValueError(f"Class '{class_name}' has 0 training samples. Add at least 1 image.")
                    
            # Extract features for all samples
            for class_idx, class_name in enumerate(self.classes):
                class_path = os.path.join(self.classes_dir, class_name)
                files = os.listdir(class_path)
                for f in files:
                    file_path = os.path.join(class_path, f)
                    try:
                        img = Image.open(file_path).convert("RGB")
                        feat = self.extract_features(img)
                        dataset_features.append(feat)
                        dataset_labels.append(class_idx)
                    except Exception as e:
                        print(f"Skipping corrupt image {file_path}: {e}")
                        
            if not dataset_features:
                raise ValueError("No valid images found for training.")
                
            # Convert dataset to tensors
            X = torch.stack(dataset_features).to(self.device)
            y = torch.tensor(dataset_labels, dtype=torch.long).to(self.device)
            
            num_samples = X.size(0)
            feat_dim = X.size(1)
            num_classes = len(self.classes)
            
            # 2. Initialize classifier
            classifier = Classifier(feat_dim, num_classes).to(self.device)
            classifier.train()
            
            optimizer = optim.Adam(classifier.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            # 3. Train
            for epoch in range(1, epochs + 1):
                # Shuffle indexes for batching
                permutation = torch.randperm(num_samples)
                epoch_loss = 0.0
                epoch_correct = 0
                
                for i in range(0, num_samples, batch_size):
                    indices = permutation[i:i + batch_size]
                    batch_x, batch_y = X[indices], y[indices]
                    
                    optimizer.zero_grad()
                    outputs = classifier(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item() * batch_x.size(0)
                    _, preds = torch.max(outputs, 1)
                    epoch_correct += torch.sum(preds == batch_y).item()
                    
                epoch_loss /= num_samples
                epoch_acc = epoch_correct / num_samples
                
                # Update status
                with self.training_lock:
                    self.training_status["current_epoch"] = epoch
                    self.training_status["progress"] = float(epoch) / epochs
                    self.training_status["metrics"]["loss"].append(epoch_loss)
                    self.training_status["metrics"]["accuracy"].append(epoch_acc)
                    
            # 4. Save model weights and finalize
            torch.save(classifier.state_dict(), self.model_path)
            
            with self.training_lock:
                self.classifier = classifier
                self.classifier.eval()
                self.training_status["status"] = "completed"
                
        except Exception as e:
            with self.training_lock:
                self.training_status["status"] = "failed"
                self.training_status["error"] = str(e)
                
    def get_training_status(self):
        """Get the current training status."""
        with self.training_lock:
            return dict(self.training_status)
            
    def predict(self, image: Image.Image):
        """Predict the class probability for a PIL Image."""
        if not self.classifier or not self.classes:
            raise RuntimeError("Model is not trained. Train the model first before predicting.")
            
        # Extract features
        feat = self.extract_features(image).unsqueeze(0).to(self.device)
        
        self.classifier.eval()
        with torch.no_grad():
            outputs = self.classifier(feat)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
            
        results = {}
        for idx, class_name in enumerate(self.classes):
            results[class_name] = float(probabilities[idx])
            
        return results

    def reset_state(self):
        """Reset the backend state by wiping classes, images, and model weights."""
        with self.training_lock:
            if self.training_status["status"] == "training":
                raise RuntimeError("Cannot reset while training is in progress.")
                
            # Clear data directory
            if os.path.exists(self.data_dir):
                shutil.rmtree(self.data_dir)
                
            os.makedirs(self.classes_dir, exist_ok=True)
            self.classes = []
            self.classifier = None
            self.training_status = {
                "status": "idle",
                "progress": 0.0,
                "current_epoch": 0,
                "total_epochs": 0,
                "metrics": {
                    "loss": [],
                    "accuracy": []
                },
                "error": None
            }
            
    def export_model_zip(self) -> io.BytesIO:
        """Export the model weights and classes.json to a zip file in memory."""
        if not os.path.exists(self.model_path) or not self.classes:
            raise RuntimeError("No trained model exists to export.")
            
        import zipfile
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            # Write classes.json
            zip_file.write(self.classes_file, "classes.json")
            # Write classifier.pt
            zip_file.write(self.model_path, "classifier.pt")
            
        zip_buffer.seek(0)
        return zip_buffer

    def import_model_zip(self, zip_bytes):
        """Import model weights and classes.json from a zip file."""
        import zipfile
        zip_buffer = io.BytesIO(zip_bytes)
        
        with zipfile.ZipFile(zip_buffer, "r") as zip_file:
            # Validate contents
            namelist = zip_file.namelist()
            if "classes.json" not in namelist or "classifier.pt" not in namelist:
                raise ValueError("Invalid model ZIP. Must contain 'classes.json' and 'classifier.pt'")
                
            # Extract to temporary variables first to avoid state corruption
            classes_content = zip_file.read("classes.json")
            model_weights_bytes = zip_file.read("classifier.pt")
            
            # Validate classes JSON
            temp_classes = json.loads(classes_content.decode("utf-8"))
            if not isinstance(temp_classes, list):
                raise ValueError("Invalid classes.json format inside ZIP.")
                
            # Reset directories and write files
            self.reset_state()
            
            with open(self.classes_file, "wb") as f:
                f.write(classes_content)
                
            with open(self.model_path, "wb") as f:
                f.write(model_weights_bytes)
                
            # Reload
            self.load_classes_metadata()
            self.load_classifier_weights()
            
            # Recreate class subfolders for persistence (even if empty, they must exist)
            for class_name in self.classes:
                os.makedirs(os.path.join(self.classes_dir, class_name), exist_ok=True)
