from ultralytics import YOLO

def main():
    # Load a pre-trained model
    model = YOLO('yolov8n.pt') 

    # Train the model 
    # Adjust epochs and imgsz as needed for your specific dataset
    results = model.train(data='data.yaml', epochs=100, imgsz=640, device='cpu') # change device to 0 if you have a compatible GPU

if __name__ == '__main__':
    main()