import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr

# Initialize the PaddleOCR reader
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Function for OCR (License Plate Text) using PaddleOCR
def extract_text_from_image(roi):
    # Convert the image to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Perform OCR using PaddleOCR
    result = ocr.ocr(gray, cls=True)
    text = " ".join([line[1][0] for line in result[0]])
    return text, gray

# Process an image for detection and OCR
def process_image(image_path, output_path):
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)

    # Extract text from the ROI
    text, processed_image = extract_text_from_image(image)
    print(f"Detected text: {text}")

    # Save the image with boxes
    cv2.imwrite(output_path, processed_image)
    print(f"Saved processed image to: {output_path}")

if __name__ == "__main__":
    # Example of image processing
    process_image("./car_plate_image.jpg", "./processed_car_plate_image.jpg")