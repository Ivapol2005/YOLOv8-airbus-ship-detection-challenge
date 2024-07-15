from ultralytics import YOLO
from PIL import Image

import os
import sys

script_directory = os.path.dirname(os.path.abspath(sys.argv[0])) 
print(script_directory)

image_dir = f"{script_directory}/test-images"
output_dir = f"{script_directory}/test-images-solved"
model_dir = f"{script_directory}/train/weights/best.pt"

model = YOLO(model_dir)

for image_file in os.listdir(image_dir):
    
    img = os.path.join(image_dir, image_file)
    input_image = Image.open(img)

    result = model(img)
    license_plate_boxes = result[0].boxes.data.cpu().numpy()

    for i, box in enumerate(license_plate_boxes):
        x1, y1, x2, y2, conf, cls = box
        license_plate = input_image.crop((x1, y1, x2, y2))
        
        license_plate.save(os.path.join(output_dir, image_file + "("+str(i)+")" + ".jpg"))


