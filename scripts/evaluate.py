from ultralytics import YOLO
def evaluate_yolov8():
    # Load the best model checkpoint from training
    model = YOLO('D:/Object_Detection/runs/detect/idcard_detection/weights/best.pt')

    # Evaluate the model on the validation dataset
    metrics = model.val(data='Dataset/data.yaml')

    print(f"Evaluation Metrics: {metrics}")

if __name__ == '__main__':
    evaluate_yolov8()