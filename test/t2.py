import cv2
import easyocr
import numpy as np

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Function for OCR (License Plate Text) using EasyOCR
def extract_text_from_image(roi):
    # Convert the image to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #_, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)
    # Perform OCR using EasyOCR
    #results = reader.readtext(gray)
    results = reader.readtext(roi)
    #results = reader.readtext(thresh)
    text = " ".join([res[1] for res in results])
    return text, roi

# Process an image for detection and OCR
def process_image(image_path, output_path):
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)

    # Extract text from the ROI
    text, processed_image = extract_text_from_image(image)
    print(f"Detected text: {text}")

    # Save the image with boxes
    #cv2.imwrite(output_path, image)
    cv2.imwrite(output_path, processed_image)
    print(f"Saved processed image to: {output_path}")

if __name__ == "__main__":
    # Example of image processing
    process_image("./car_plate_image.jpg", "./processed_car_plate_image.jpg")