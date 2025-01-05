import os
import glob
import torch
import cv2
import logging
from torchvision import models, transforms
from torchvision.ops import nms
#from torchvision.models import  resnet50, ResNet50_Weights
from PIL import Image
from ultralytics import YOLO
import numpy as np
from paddleocr import PaddleOCR #, draw_ocr

# Set logging level to suppress YOLOv5 messages
logging.getLogger('ultralytics').setLevel(logging.WARNING)
# Set logging level to suppress PaddleOCR debug messages
logging.getLogger('ppocr').setLevel(logging.WARNING)

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
#def detect_objects_cars(image_path, iou_threshold=0.5):
def detect_objects_cars(image_cv2, iou_threshold=0.5):
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)

    #image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        results = model_cars(image_tensor)

    # Extract bounding boxes, confidence scores, and class labels
    predictions = results[0]
    boxes = predictions['boxes']
    scores = predictions['scores']
    labels = predictions['labels']

    # Apply Non-Maximum Suppression (NMS)
    keep = nms(boxes, scores, iou_threshold)
    boxes = boxes[keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()
    labels = labels[keep].cpu().numpy()
    
    #return prediction
    return {"boxes": boxes, "scores": scores, "labels": labels}

# Function to detect objects (license plates)
def detect_objects_plates(roi, iou_threshold=0.5):
    results = model_plates(roi)
    predictions = results[0]  # Access the first element of the list

    # Extract bounding boxes, confidence scores, and class labels
    boxes = predictions.boxes.xyxy
    scores = predictions.boxes.conf
    labels = predictions.boxes.cls

    # Apply Non-Maximum Suppression (NMS)
    keep = nms(boxes, scores, iou_threshold)
    boxes = boxes[keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()
    labels = labels[keep].cpu().numpy()

    return {"boxes": boxes, "scores": scores, "labels": labels}

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
    image = cv2.imread(image_path)

    # Data structure to store car, plate, and text data
    car_data = []

    #results_cars = detect_objects_cars(image_path)
    results_cars = detect_objects_cars(image)
    boxes = results_cars['boxes']

    #image = cv2.imread(image_path)
    for box_car in boxes:
        c_x1, c_y1, c_x2, c_y2 = box_car.astype(int)

        # Extract the region of interest (ROI) for the car
        roi_car = image[c_y1:c_y2, c_x1:c_x2]            

        # Check if the ROI has non-zero dimensions
        if roi_car.shape[0] == 0 or roi_car.shape[1] == 0:
            print(f"Skipping empty ROI for car: {box_car}")
            continue

        print(f"Detected car box: {box_car}")
        #c_text = f"Detected car box: {box_car}"

        results_plates = detect_objects_plates(roi_car)
        boxes_plates = results_plates['boxes']
        for box_plate in boxes_plates:
            print(f"Detected plate box: {box_plate}")
            #p_text = f"Detected license plate text: {box_plate}"
            bp_x1, bp_y1, bp_x2, bp_y2 = box_plate.astype(int)

            # Adjust coordinates relative to the original image
            p_x1, p_y1, p_x2, p_y2 = bp_x1 + c_x1, bp_y1 + c_y1, bp_x2 + c_x1, bp_y2 + c_y1

            # Extract the region of interest (ROI) for the number plate
            roi_plate = image[p_y1:p_y2, p_x1:p_x2]            

            # Extract text from the ROI
            text, thresh = extract_text_from_image(roi_plate)

            if text != "":
                print(f"Detected license plate text: {text}")

                # Initialize car entry
                car_entry = {
                    "car_box": box_car.tolist(),
                    "plates": []
                }

                # Add plate data to car entry
                car_entry["plates"].append({
                    "plate_box": box_plate.tolist(),
                    "text": text
                })

                # Add car entry to car data
                car_data.append(car_entry)

    # Sometimes a car box overlaps more than one car and this results in there being more that one plate in the car box roi
    #
    # Need to flag if more than one plate is detected and then determine which one is the correct one.
    # this will need to be done by creating a list of the cars and the plates detected in each car box.
    # we can then compare the plates detected in each car box to the plates detected in other car boxes.
    # a car with only one plate detected will likey have the correct plate assigned to it.
    # if a car has more than one plate detected then we can compare the plates detected in that car box to the plates detected in other car boxes.
    # this can then be used to remove the false positives.
    # we will need to set up a suitable data structure to store the car, plate, and text data.
    # we will only add cars that have plates with text to the data structure.

    # Process car data to remove false positives
    for car in car_data:
        #car_box = car["car_box"]
        plates = car["plates"]
        if len(plates) > 1:
            # Compare plates with other cars
            for other_car in car_data:
                if other_car == car:
                    continue
                #other_car_box = other_car["car_box"]
                other_plates = other_car["plates"]
                for plate in plates:
                    plate_text = plate["text"]
                    for other_plate in other_plates:
                        other_plate_text = other_plate["text"]
                        # Check if the plate is a false positive
                        if np.array_equal(plate_text, other_plate_text):
                            print(f"Removing false positive plate: {plate}")
                            plates.remove(plate)
                            break

    # Process car data to draw bounding boxes and text
    for car in car_data:
        box_car = car["car_box"]
        plates = car["plates"]
        c_x1, c_y1, c_x2, c_y2 = box_car.astype(int)
        for plate in plates:
            box_plate = plate["plate_box"]
            text = plate["text"]
            bp_x1, bp_y1, bp_x2, bp_y2 = box_plate.astype(int)
            p_x1, p_y1, p_x2, p_y2 = bp_x1 + c_x1, bp_y1 + c_y1, bp_x2 + c_x1, bp_y2 + c_y1

            # Extract the region of interest (ROI) for the number plate
            cv2.rectangle(image, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 0), 2) # Draws CAR bounding box
            cv2.rectangle(image, (p_x1, p_y1), (p_x2, p_y2), (0, 255, 0), 2) # Draws PLATE bounding box inside the car bounding box
            cv2.putText(image, text, (p_x1, p_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2) # Draws PLATE text above the plate bounding box

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