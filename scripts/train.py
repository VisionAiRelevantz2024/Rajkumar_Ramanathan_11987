from ultralytics import YOLO

def train_yolov8():
    # Load the YOLOv8 model (choose YOLOv8n or YOLOv8s)
    model = YOLO('yolov8s.pt')  # or yolov8s.pt depending on model size

    # Train the model
    model.train(
        data='D:\Object_Detection\Dataset\data.yaml',  # Path to the dataset YAML
        epochs=50,  # Number of epochs
        batch=16,  # Batch size
        imgsz=640,  # Image size
        save_period=10,  # Save checkpoint every 10 epochs
        name='idcard_detection'  # Name of the project for saving outputs
    )

if __name__ == '__main__':
    train_yolov8()