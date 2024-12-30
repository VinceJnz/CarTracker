import torch
import cv2
import pytesseract
from torchvision import models, transforms
from PIL import Image


# Function for OCR (License Plate Text)
def extract_text_from_image(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #_, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    text = pytesseract.image_to_string(thresh, config='--psm 6')
    return text, thresh

# Process an image for detection and OCR
def process_image(image_path, output_path):
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)

    # Extract text from the bounding box
    text, thresh = extract_text_from_image(image)
    print(f"Detected license plate text: {text}")
        
    # Get bounding boxes for each character
    boxes = pytesseract.image_to_boxes(thresh, config='--psm 6')
    #h, w, _ = image.shape
    h, w = thresh.shape
    for b in boxes.splitlines():
        b = b.split(' ')
        tx1, ty1, tx2, ty2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        #cv2.rectangle(image, (tx1, h - ty2), (tx2, h - ty1), (255, 0, 0), 2)
        cv2.rectangle(thresh, (tx1, h - ty2), (tx2, h - ty1), (255, 0, 0), 2)
    
    # Save the image with boxes
    #cv2.imwrite(output_path, image)
    cv2.imwrite(output_path, thresh)
    print(f"Saved processed image to: {output_path}")

if __name__ == "__main__":
    # Example of image processing
    process_image("./car_plate_image.jpg", "./processed_car_plate_image.jpg")