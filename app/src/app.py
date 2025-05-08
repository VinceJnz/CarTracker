import os
#import glob
from pathlib import Path
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

TestImageCount = 0

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
    #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Perform OCR using PaddleOCR
    #result = ocr.ocr(gray, cls=True)
    result = ocr.ocr(roi, cls=True)
    if result == [None]:
        #return "", gray, 0
        return "", 0
    text = " ".join([line[1][0] for line in result[0]])
    confidence = min([line[1][1] for line in result[0]])
    #return text, gray, confidence
    return text, confidence

# Process an image for detection and OCR
def process_image(image, output_path=""):
    global TestImageCount
    print(f"Processing image started")

    # Data structure to store car, plate, and text data
    car_data = []

    #results_cars = detect_objects_cars(image_path)
    results_cars = detect_objects_cars(image) #???????????? Need to update this to be able to use confidences in the output. ????????????
    boxes = results_cars['boxes']
    scores = results_cars['scores']
    labels = results_cars['labels']
    print(f"results_cars: {results_cars}, boxes: {boxes}, scores: {scores}, labels: {labels}")

    #image = cv2.imread(image_path)
    for car_index, (car_box, car_score) in enumerate(zip(boxes, scores)):
        c_x1, c_y1, c_x2, c_y2 = car_box.astype(int)

        # Extract the region of interest (ROI) for the car
        roi_car = image[c_y1:c_y2, c_x1:c_x2]            

        # Check if the ROI has non-zero dimensions
        if roi_car.shape[0] == 0 or roi_car.shape[1] == 0:
            #print(f"Skipping empty ROI for car: {box_car}")
            continue

        results_plates = detect_objects_plates(roi_car)
        boxes_plates = results_plates['boxes']
        for plate_index, box_plate in enumerate(boxes_plates):
            bp_x1, bp_y1, bp_x2, bp_y2 = box_plate.astype(int)

            # Adjust coordinates relative to the original image
            p_x1, p_y1, p_x2, p_y2 = bp_x1 + c_x1, bp_y1 + c_y1, bp_x2 + c_x1, bp_y2 + c_y1

            # Extract the region of interest (ROI) for the number plate
            roi_plate = image[p_y1:p_y2, p_x1:p_x2]            

            # Extract text from the ROI
            text, confidence = extract_text_from_image(roi_plate)

            if text != "":
                #print(f"Detected license plate text: {text}")

                # Initialize car entry
                if plate_index == 0:
                    car_entry = {
                        "car_id": car_index,
                        "car_box": car_box, #.tolist(),
                        "car_score": car_score,
                        "plates": []
                    }

                # Add plate data to car entry
                car_entry["plates"].append({
                    "plate_box": box_plate, #.tolist(),
                    "text": text,
                    "text_confidence": round(confidence, 5)
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
    #print(f"Starting review of car data/plate")
    #for car in car_data:
    #    # Reviewing car
    #    plates = car["plates"]
    #    print(f"plates: {plates}", type(plates))
    #    if len(plates) > 1:
    #        # Create a new list to store plates to keep
    #        #plates_to_keep = []
    #        plates_to_keep = plates.copy() # Create a copy of the plates list, need to use copy() to avoid reference to the original list

    #        for plate in plates:
    #            plate_text = plate["text"]
    #            plate_confidence = plate["text_confidence"]

    #            # Compare plates with other cars
    #            for other_car in car_data:
    #                # Reviewing other car
    #                if np.array_equal(other_car["car_box"], car["car_box"]):
    #                    continue
    #                other_plates = other_car["plates"]

    #            #for plate in plates:
    #            #    plate_text = plate["text"]
    #                for other_plate in other_plates:
    #                    other_plate_text = other_plate["text"]
    #                    other_plate_confidence = other_plate["text_confidence"]
    #                    # Check if the plate is a false positive
    #                    if plate_text == other_plate_text: # and plate_confidence < other_plate_confidence:
    #                        # Removing false positive plate
    #                        plates_to_keep.remove(plate)
    #                        break
    #        car["plates"] = plates_to_keep
    #        print(f"updated plates: {car['plates']}", type(car['plates']))


    print(f"Starting review of car/plate data")
    for car in car_data:
        plates = car["plates"]
        print(f"plates: {plates}, plates type: {type(plates)}, car type: {type(car)}")
        if len(plates) > 1:
            # Create a new list to store plates to keep
            plates_to_keep = []
            # Create a dictionary to store the highest confidence plate for each text
            #highest_confidence_plates = {}
            for plate in plates:
                plate_text = plate["text"]
                is_false_positive = False
                for other_car in car_data:
                    if np.array_equal(other_car["car_box"], car["car_box"]):
                        continue
                    other_plates = other_car["plates"]
                    for other_plate in other_plates:
                        other_plate_text = other_plate["text"]
                        if plate_text == other_plate_text:
                           is_false_positive = True
                           break
                    if is_false_positive:
                        break
                if not is_false_positive:
                    #if plate_text not in highest_confidence_plates or plate["text_confidence"] > highest_confidence_plates[plate_text]["text_confidence"]:
                    #    highest_confidence_plates[plate_text] = plate
                    plates_to_keep.append(plate)
            # Replace the original plates list with the filtered list
            car["plates"] = plates_to_keep
            #car["plates"] = list(highest_confidence_plates.values())
            print(f"updated plates: {car['plates']}", type(car['plates']))
    
    print(f"car_data: {car_data}, type {type(car_data)}")
    # Process car data to keep only the plate with the highest confidence for each unique plate text
    print(f"Starting review of car data")
    # Create a dictionary to store the highest confidence plate for each text
    highest_confidence_car = {}
    for car in car_data:
        plates = car["plates"]
        print(f"plates: {plates}", type(plates))
        for plate1 in plates:
            print(f"plate1: {plate1}", type(plate1))
            plate1_text = plate1["text"]
            if plate1_text not in highest_confidence_car:
                highest_confidence_car[plate1_text] = car
            else:
                print(f"plate1_text: {plate1_text}, highest_confidence_car: {highest_confidence_car[plate1_text]}")
                for plate2 in highest_confidence_car[plate1_text]["plates"]:
                    if plate2["text_confidence"] > plate1["text_confidence"]:
                        highest_confidence_car[plate1_text] = car
                        break
        # Replace the original plates list with the filtered list
    car_data = list(highest_confidence_car.values())
    print(f"updated car_data: {car_data}, type {type(car_data)}")

    # Process car data to draw bounding boxes and text
    print(f"Starting drawing boxes and text")
    # Debugging
    image1 = image.copy()
    for car in car_data:
        c_x1, c_y1, c_x2, c_y2 = car["car_box"].astype(int)
        for plate in car["plates"]:
            box_plate = plate["plate_box"]
            text_1 = "car: " + str(car["car_id"]) +", confidence: "+ str(car["car_score"])
            text_2 =  "plate: " + plate["text"] + ", confidence: " + str(plate["text_confidence"])
            bp_x1, bp_y1, bp_x2, bp_y2 = box_plate.astype(int)
            p_x1, p_y1, p_x2, p_y2 = bp_x1 + c_x1, bp_y1 + c_y1, bp_x2 + c_x1, bp_y2 + c_y1

            # Extract the region of interest (ROI) for the number plate
            cv2.rectangle(image, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 0), 2) # Draws CAR bounding box
            cv2.rectangle(image, (p_x1, p_y1), (p_x2, p_y2), (0, 255, 0), 2) # Draws PLATE bounding box inside the car bounding box
            cv2.putText(image, text_1, (p_x1, p_y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2) # Draws PLATE text above the plate bounding box
            cv2.putText(image, text_2, (p_x1, p_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2) # Draws PLATE text above the plate bounding box

            # Debugging - Extract the region of interest (ROI) for the number plate
            image2 = image1.copy()
            cv2.rectangle(image2, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 0), 2) # Draws CAR bounding box
            cv2.rectangle(image2, (p_x1, p_y1), (p_x2, p_y2), (0, 255, 0), 2) # Draws PLATE bounding box inside the car bounding box
            cv2.putText(image2, text_1, (p_x1, p_y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2) # Draws PLATE text above the plate bounding box
            cv2.putText(image2, text_2, (p_x1, p_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2) # Draws PLATE text above the plate bounding box
            TestImageCount += 1
            fileName = "../data/processed/ImageAnalysis/Image" + str(TestImageCount) + ".jpg"
            print(f"Saving processed image to: {fileName}")
            cv2.imwrite(fileName, image2)

    # Save the image with boxes
    if output_path!="":
        cv2.imwrite(output_path, image)
        print(f"Saved processed image to: {output_path}")

    print(f"Image processing finished\n")
    return image
  

# Function to get a list of files with a case-insensitive glob pattern
def case_insensitive_file_list(path, pattern):
    input_path = Path(path)
    files = [str(f) for f in input_path.iterdir() if f.is_file() and f.name.lower().endswith(pattern)]
    #files = [str(f) for f in input_path.iterdir() if f.is_file() and f.name.lower().endswith(".mp4")]
    return files


# Process video frames
def process_videos(input_path, output_path, frame_gap=20, rotate=False):
    print(f"Processing videos from: {input_path}, to {output_path}\n")
    os.makedirs(output_path, exist_ok=True)

    # Use the case_insensitive function to get the input files
    input_files = case_insensitive_file_list(input_path, ".mp4")
    print(f"video file path list: {input_files}")

    # Process each image file
    for input_file_path in input_files:
        input_file_name = os.path.basename(input_file_path)
        output_file_path = os.path.join(output_path, f"processed_{input_file_name}")

        print(f"video file paths: {input_file_path}, {output_file_path}")

        cap = cv2.VideoCapture(input_file_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_file_path}")
            return
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        frame_num = 0
        next_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num != next_frame:
                frame_num += 1
                continue

            next_frame = frame_num + frame_gap
            print(f"processing frame: {frame_num}")
            # Process the frame
            #output_image_file_path = os.path.join(output_path, f"processed_{input_file_name}_{str(frame_num)}.jpg")

            # Add this line in if the frame is upside down and needs to be rotated.
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_180) # Rotate the frame 180 degrees only if needed.
            
            processed_frame = process_image(frame)

            # Check if processed_frame is None
            if processed_frame is None:
                print(f"Error processing frame {frame_num}")
                frame_num += 1
                continue

            #height, width, channels = processed_frame.shape
            #size = processed_frame.size
            #print(f"Processed frame shape: Height: {height}, Width: {width}, Channels: {channels}")
            #print(f"Processed frame size: {size} pixels")

            # Write the processed frame to the output video
            out.write(processed_frame)
            frame_num += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()
    print(f"Processing videos finished\n")
    

def rotate_video(input_path, output_path, angle):
    print(f"Processing videos from: {input_path}, to {output_path}\n")
    os.makedirs(output_path, exist_ok=True)

    # Use the case_insensitive function to get the input files
    input_files = case_insensitive_file_list(input_path, ".mp4")
    print(f"video file path list: {input_files}")

    # Define rotation mappings based on the angle
    rotation_mapping = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE
    }

    # Check if the angle is valid
    if angle not in rotation_mapping:
        print(f"Error: Unsupported rotation angle {angle}. Supported angles are 90, 180, and 270.")
        return

    # Get the appropriate rotation code
    rotation_code = rotation_mapping[angle]

    # Process each image file
    for input_file_path in input_files:
        input_file_name = os.path.basename(input_file_path)
        output_file_path = os.path.join(output_path, f"processed_{input_file_name}")

        print(f"video file paths: {input_file_path}, {output_file_path}")

        cap = cv2.VideoCapture(input_file_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_file_path}")
            return
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            
            frame = cv2.rotate(frame, rotation_code) # Rotate the frame 180 degrees only if needed.

            # Check if processed_frame is None
            if frame is None:
                print(f"Error processing frame {frame_num}")
                continue

            # Write the processed frame to the output video
            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
    print(f"Processing videos finished\n")





# Process video frames
def process_images(input_path, output_path):
    print(f"Processing images from: {input_path}, to {output_path}\n")
    os.makedirs(output_path, exist_ok=True)

    # Define the list of file extensions
    extensions = [".jpg", ".jpeg", ".png"]

    # Collect all files with the specified extensions
    input_image_files = []
    for ext in extensions:
        input_image_files.extend(case_insensitive_file_list(input_path, ext))
        #input_image_files.extend(glob.glob(os.path.join(input_path, ext)))

    # Process each image file
    for input_image_file in input_image_files:
        input_file_name = os.path.basename(input_image_file)
        output_file_name = os.path.join(output_path, f"processed_{input_file_name}")
        print(f"Processing image: {input_file_name}, to {output_file_name}")
        image = cv2.imread(input_image_file)
        process_image(image, output_file_name)


if __name__ == "__main__":
    print(f"Processing has started\n")

    input_folder = "../data"
    output_folder = "../data/processed"
    #process_videos(input_folder, output_folder, 20*40, rotate=True)
    process_videos(input_folder, output_folder, 10, rotate=False)

    process_images(input_folder, output_folder)

