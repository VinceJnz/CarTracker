import os
import glob
import torch
import cv2
import logging
from torchvision import models, transforms
from torchvision.models import  resnet50, ResNet50_Weights
from PIL import Image
from ultralytics import YOLO
import numpy as np
from paddleocr import PaddleOCR, draw_ocr

# Reference: License plate detection using YOLOv8
#https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/blob/main/README.md

# Set logging level to suppress YOLOv5 messages
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Initialize the PaddleOCR reader
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load the ResNet model
# https://pytorch.org/vision/stable/models.html
model_cars = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#weights = ResNet50_Weights.DEFAULT #model_cars = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
#preprocess_cars = weights.transforms()
#model_cars = resnet50(weights=weights)
model_cars.eval()

#Type of results_cars: <class 'list'>
#Length of results_cars: 1
#Keys in results_cars[0]: dict_keys(['boxes', 'labels', 'scores'])
#Shape of results_cars[0]['boxes']: torch.Size([68, 4])

#Type of results_cars: <class 'torch.Tensor'>
#Shape of results_cars: torch.Size([1000])

# Load the YOLOv5 model for license plate detection
model_plates = YOLO('../models/YOLO_license_plate_detector.pt')
model_plates.eval()

# Image transformation for YOLOv5
transform = transforms.Compose([transforms.ToTensor()])

# Function to detect objects (cars)
def detect_objects_cars(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    #image_tensor = preprocess_cars(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model_cars(image_tensor)
    #prediction = model_cars(image_tensor).squeeze(0) #.softmax(0)
    
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

    # debug stuff
    print(f"Type of results_cars: {type(results_cars)}")
    if isinstance(results_cars, torch.Tensor):
        print(f"Shape of results_cars: {results_cars.shape}")
    elif isinstance(results_cars, list):
        print(f"Length of results_cars: {len(results_cars)}")
        if len(results_cars) > 0 and isinstance(results_cars[0], dict):
            print(f"Keys in results_cars[0]: {results_cars[0].keys()}")
            if 'boxes' in results_cars[0]:
                print(f"Shape of results_cars[0]['boxes']: {results_cars[0]['boxes'].shape}")
    
    image = cv2.imread(image_path)
    for box_car in results_cars[0]['boxes']:
        c_x1, c_y1, c_x2, c_y2 = box_car.int().numpy()
        #cv2.rectangle(image, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 0), 2) # Draws bounding box
        #print(f"Detected car box: {box_car}")

        # Extract the region of interest (ROI) for the car
        roi_car = image[c_y1:c_y2, c_x1:c_x2]            
        #output_path_temp = "./processed_car_image_temp1.jpg"
        #cv2.imwrite(output_path_temp, roi_car)
        #print(f"Saved processed image to: {output_path_temp}")

        #results_plates = detect_objects_plates(image_path, (c_x1, c_y1, c_x2, c_y2))
        results_plates = detect_objects_plates(roi_car)
        for license_plate in results_plates.boxes.data.tolist():
            p_x1, p_y1, p_x2, p_y2, score, class_id = license_plate
            box_plate = (p_x1+c_x1, p_y1+c_y1, p_x2+c_x1, p_y2+c_y1)

            x1, y1, x2, y2 = np.array(box_plate).astype(int)
            #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Draws bounding box
            #print(f"Detected plate box: {box_plate}")

            # Extract the region of interest (ROI) for the number plate
            roi_plate = image[y1:y2, x1:x2]            
            #output_path_temp = "./processed_car_image_temp2.jpg"
            #cv2.imwrite(output_path_temp, roi_plate)
            #print(f"Saved processed image to: {output_path_temp}")

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
    #process_image("../data/1car_image.jpg", "../data/1processed_car_image.jpg")
    #process_image("../data/2car_image.jpg", "../data/2processed_car_image.jpg")
    #process_image("../data/3car_image.jpg", "../data/3processed_car_image.jpg")
    #process_image("../data/4car_image.png", "../data/4processed_car_image.jpg")

    input_folder = "../data"
    output_folder = "../data/processed"
    os.makedirs(output_folder, exist_ok=True)
    input_image_files = glob.glob(os.path.join(input_folder, "*.*"))

    # Process each image file
    for input_image_file in input_image_files:
        input_file_name = os.path.basename(input_image_file)
        output_impage_file = os.path.join(output_folder, f"processed_{input_file_name}")
        process_image(input_image_file, output_impage_file)