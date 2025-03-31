from ultralytics import YOLO
from pathlib import Path
import torch
import os

device = '0' if torch.cuda.is_available() else 'cpu'
print(f"\n🚀 Training on {'GPU' if device != 'cpu' else 'CPU'}")
# Training configuration
train_args = {
        # Data configuration
        "data": str(Path("data/config.yaml").absolute()),
        "epochs": 300,
        "batch": 8,
        "imgsz": 640,
        "device": device,
        "workers": min(4, os.cpu_count() - 1),
        "seed": 42,
        "split": 0.2,  # Auto 80/20 split
        
        # Learning parameters
        "lr0": 0.001,
        "lrf": 0.01,
        "warmup_epochs": 10,
        "weight_decay": 0.005,
        
        # UI-specific augmentations
        "augment": True,
        "hsv_h": 0.015,
        "hsv_s": 0.2,  # Reduced for UI color consistency
        "hsv_v": 0.3,
        "degrees": 2,  # Minimal rotation
        "fliplr": 0.02,
        "mosaic": 0.8,
        "mixup": 0.1,
        
        # Regularization
        "dropout": 0.3,
        "label_smoothing": 0.1,
        "patience": 50,
        
        # Logging
        "save": True,
        "save_period": 10,
        "plots": True,
        "exist_ok": True  # Overwrite existing runs
    }

model = YOLO("yolov8n.yaml")
results = model.train(
    # data="config.yaml", 
    # epochs=300,  # More epochs to squeeze learning from limited data
    # imgsz=640,
    # patience=50,  # Early stopping
    # batch=4,  # Smaller batch due to limited images
    # augment=True  # Enable data augmentation
    **train_args
)