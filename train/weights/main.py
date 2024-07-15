from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("/home/cash/runs/detect/train/weights/best.pt")

# Define path to the image file
source = "/home/cash/Документи/model_ship/test/images/7f62e00d-0a0df8299.jpg"

# Run inference on the source
results = model(source)  # list of Results objects
