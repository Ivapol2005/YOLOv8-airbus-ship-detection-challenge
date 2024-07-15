from ultralytics import YOLO
import os
from PIL import Image


current_dir = os.getcwd()

image_dir = os.path.join(current_dir,"model/test-images/images")
output_dir = os.path.join(current_dir,"model/test-images-solved")

model = YOLO(os.path.join(current_dir,"model/train/weights/best.pt"))

for image_file in os.listdir(image_dir):
    
    img = os.path.join(image_dir, image_file)
    input_image = Image.open(img)

    result = model(img)
    license_plate_boxes = result[0].boxes.data.cpu().numpy()

    for i, box in enumerate(license_plate_boxes):
        x1, y1, x2, y2, conf, cls = box
        license_plate = input_image.crop((x1, y1, x2, y2))
        
        license_plate.save(os.path.join(output_dir, "("+str(i)+")" + image_file))


