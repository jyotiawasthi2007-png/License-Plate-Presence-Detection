import cv2
from ultralytics import YOLO
import os

def run_inference(source, weights='runs/detect/train/weights/best.pt', conf=0.25):
    if not os.path.exists(weights):
        print(f"Weights not found at {weights}. Using default yolov8m.pt")
        weights = 'yolov8m.pt'
        
    model = YOLO(weights)
    
    # Run inference
    results = model.predict(
        source=source,
        conf=conf,
        iou=0.45,
        device='0',      # Use GPU
        save=True,       # Save results
        show=False,      # Don't show window (good for scripts)
        line_width=2,
        augment=True     # Use TTA for better accuracy
    )
    
    print(f"Inference completed. Results saved in {results[0].save_dir}")
    
    # Process results
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf_score = float(box.conf[0])
            label = model.names[cls]
            print(f"Detected: {label} with confidence {conf_score:.2f}")

if __name__ == "__main__":
    # Test on a sample image from the test set
    test_img_dir = r'D:\License-Plate-Detection-main\YOLO_Dataset\test\images'
    if os.path.exists(test_img_dir):
        sample_img = os.path.join(test_img_dir, os.listdir(test_img_dir)[0])
        print(f"Testing on sample: {sample_img}")
        run_inference(sample_img)
    else:
        print("Test dataset not found. Please provide a valid source.")
