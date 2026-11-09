
from ultralytics import YOLO
import os

# Dataset configuration
DATA_YAML = r"D:\AquaVision\Dataset\raw\underwater_trash_detection\data.yaml"

# Create results directory
os.makedirs("results", exist_ok=True)

print("=" * 50)
print("       AQUAVISION MODEL TRAINING")
print("=" * 50)

# Load YOLOv8 Nano pretrained model
model = YOLO("yolov8n.pt")

print("\nStarting model training...")

# Train model

results = model.train(
    data=DATA_YAML,
    epochs=5,
    imgsz=320,
    batch=2,
    device="cpu",
    workers=0,
    fraction=0.25,
    project="results",
    name="aquavision_quick",
    patience=3,
    plots=True
)


print("\nTraining completed!")

print("Best model saved at:")
print("results/aquavision_yolov8/weights/best.pt")

print("\nAquaVision training finished!")
