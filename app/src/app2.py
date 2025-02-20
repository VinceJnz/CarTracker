import torch
import cv2
import pytesseract
from torchvision import models, transforms
from PIL import Image

# Load the pre-trained Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Image transformation for Faster R-CNN
transform = transforms.Compose([transforms.ToTensor()])

# Function to detect objects (cars)
def detect_objects(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(image_tensor)
    
    return prediction

# Function for OCR (License Plate Text)
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, config='--psm 6')
    return text, thresh

# Process an image for detection and OCR
def process_image(image_path, output_path):
    print(f"Processing image: {image_path}")
    prediction = detect_objects(image_path)
    
    image = cv2.imread(image_path)
    for box in prediction[0]['boxes']:
        x1, y1, x2, y2 = box.int().numpy()
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print(f"Detected box: {box}")
    
    text, thresh = extract_text_from_image(image_path)
    print(f"Detected license plate text: {text}")
    
    # Get bounding boxes for each character
    boxes = pytesseract.image_to_boxes(thresh, config='--psm 6')
    h, w, _ = image.shape
    for b in boxes.splitlines():
        b = b.split(' ')
        x1, y1, x2, y2 = int(b[1]), h - int(b[2]), int(b[3]), h - int(b[4])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Save the image with boxes
    cv2.imwrite(output_path, image)
    print(f"Saved processed image to: {output_path}")

if __name__ == "__main__":
    # Example of image processing
    process_image("data/car_image.jpg", "data/processed_car_image.jpg")