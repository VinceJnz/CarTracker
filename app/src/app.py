import os
import glob
import torch
import cv2
import logging
from torchvision import models, transforms
#from torchvision.models import  resnet50, ResNet50_Weights
from PIL import Image
from ultralytics import YOLO
import numpy as np
from paddleocr import PaddleOCR #, draw_ocr

# Set logging level to suppress YOLOv5 messages
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Initialize the PaddleOCR reader
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load the ResNet model
# https://pytorch.org/vision/stable/models.html
model_cars = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_cars.eval()

# Load the YOLOv5 model for license plate detection
# Reference: License plate detection using YOLOv8
# https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/blob/main/README.md
model_plates = YOLO('../models/YOLO_license_plate_detector.pt')
model_plates.eval()

# Image transformation for YOLOv5
transform = transforms.Compose([transforms.ToTensor()])

# Function to detect objects (cars)
def detect_objects_cars(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        prediction = model_cars(image_tensor)
    return prediction

# Function to detect objects (license plates)
def detect_objects_plates(roi):
    results = model_plates(roi)
    return results[0]

# Function for OCR (License Plate Text) using PaddleOCR
def extract_text_from_image(roi):
    # Convert the image to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Perform OCR using PaddleOCR
    result = ocr.ocr(gray, cls=True)
    if result == [None]:
        return "", gray
    text = " ".join([line[1][0] for line in result[0]])
    return text, gray

# Process an image for detection and OCR
def process_image(image_path, output_path):
    print(f"Processing image: {image_path}")
    results_cars = detect_objects_cars(image_path)

    image = cv2.imread(image_path)
    for box_car in results_cars[0]['boxes']:
        c_x1, c_y1, c_x2, c_y2 = box_car.int().numpy()
        #cv2.rectangle(image, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 0), 2) # Draws bounding box
        #print(f"Detected car box: {box_car}")

        # Extract the region of interest (ROI) for the car
        roi_car = image[c_y1:c_y2, c_x1:c_x2]            

        results_plates = detect_objects_plates(roi_car)
        for license_plate in results_plates.boxes.data.tolist():
            p_x1, p_y1, p_x2, p_y2, score, class_id = license_plate
            box_plate = (p_x1+c_x1, p_y1+c_y1, p_x2+c_x1, p_y2+c_y1)

            x1, y1, x2, y2 = np.array(box_plate).astype(int)

            # Extract the region of interest (ROI) for the number plate
            roi_plate = image[y1:y2, x1:x2]            

            # Extract text from the ROI
            text, thresh = extract_text_from_image(roi_plate)
            print(f"Detected license plate text: {text}")

            if text != "":
                cv2.rectangle(image, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 0), 2) # Draws CAR bounding box
                print(f"Detected car box: {box_car}")
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Draws PLATE bounding box
                print(f"Detected plate box: {box_plate}")
                cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Save the image with boxes
    cv2.imwrite(output_path, image)
    print(f"Saved processed image to: {output_path}")

if __name__ == "__main__":
    # Example of image processing
    #process_image("../data/car_image.jpg", "../data/processed_car_image.jpg")

    input_folder = "../data"
    output_folder = "../data/processed"
    os.makedirs(output_folder, exist_ok=True)
    input_image_files = glob.glob(os.path.join(input_folder, "*.*"))

    # Process each image file
    for input_image_file in input_image_files:
        input_file_name = os.path.basename(input_image_file)
        output_impage_file = os.path.join(output_folder, f"processed_{input_file_name}")
        process_image(input_image_file, output_impage_file)