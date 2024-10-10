import cv2
import easyocr  # For OCR
import pymongo  # For MongoDB
from ultralytics import YOLO
from datetime import datetime

# MongoDB configuration (update URI as per your setup)
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["idcard_db"]
collection = db["idcard_collection"]

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])  # You can add more languages if needed

# Function to extract name and employee ID from OCR text
def parse_ocr_text(ocr_results):
    name = None
    emp_id = None
    for result in ocr_results:
        text = result[1].strip()  # Extract the recognized text

        # Detect the employee ID (assuming it includes 'No' or 'Employee')
        if 'no' in text.lower() or 'employee' in text.lower():
            emp_id = text.split()[-1]  # Get the last word (the number)
        
        # Detect the name (assuming it does not contain 'Employee' or 'No')
        elif not any(keyword in text.lower() for keyword in ['no', 'employee', 'blood', 'group']):
            name = text

        # Stop once both name and employee ID are found
        if name and emp_id:
            break
    
    return name, emp_id

# Function to store the details in MongoDB
def store_in_mongodb(name, emp_id, timestamp):
    record = {"name": name, "emp_id": emp_id, "timestamp": timestamp}
    collection.insert_one(record)
    print(f"Stored in MongoDB: {record}")

def predict_from_camera():
    # Load the trained model
    model = YOLO('D:/Object_Detection/runs/detect/idcard_detection/weights/best.pt')

    # Open the laptop camera (camera index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        # Capture frame-by-frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame.")
            break

        # Predict using YOLO model
        results = model.predict(source=frame, show=False)

        # Visualize the results
        annotated_frame = results[0].plot()  # Get the annotated frame

        # Display the frame with predictions
        cv2.imshow('ID Card Detection', annotated_frame)

        # Check if ID card is detected
        if len(results[0].boxes) > 0:
            print(f"ID Card detected with confidence scores: {[box.conf for box in results[0].boxes]}")
            
            # Extract the bounding box of the detected ID card
            for box in results[0].boxes:
                # Get the bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Crop the detected ID card area
                id_card_area = frame[y1:y2, x1:x2]

                # Run OCR on the detected ID card area
                ocr_results = reader.readtext(id_card_area)
                print(f"OCR Results: {ocr_results}")

                # Parse the OCR text to extract name and employee ID
                name, emp_id = parse_ocr_text(ocr_results)

                if name and emp_id:
                    print(f"ID Card found - Name: {name}, Employee ID: {emp_id}")
                    
                    # Store the result in MongoDB with a timestamp
                    timestamp = datetime.now()
                    store_in_mongodb(name, emp_id, timestamp)
                else:
                    print("ID Card found, but failed to extract details.")
        else:
            print("No ID Card found.")

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    predict_from_camera()