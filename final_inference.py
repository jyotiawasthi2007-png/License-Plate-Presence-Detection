from ultralytics import YOLO
import os
import glob
import cv2
import matplotlib.pyplot as plt

def run_inference():
    # Load best model
    # Note: Replace with actual path if different
    model_path = 'runs/detect/license_plate_dual_class/weights/best.pt'
    if not os.path.exists(model_path):
        # Fallback to current best if not yet trained
        model_path = 'yolov8n.pt'
        print(f"Warning: Custom model not found at {model_path}. Using pre-trained weights for demo.")
    
    model = YOLO(model_path)
    
    # Test images path
    test_images = glob.glob('Final_YOLO_Dataset/test/images/*.jpg')
    
    if not test_images:
        print("No test images found.")
        return

    # Create output dir
    os.makedirs('inference_results', exist_ok=True)

    summary = {0: 0, 1: 0}
    total_images = len(test_images)

    for img_path in test_images:
        results = model.predict(source=img_path, conf=0.25, save=False, verbose=False)
        
        detections = []
        for r in results:
            for c in r.boxes.cls:
                cls_id = int(c)
                summary[cls_id] += 1
                detections.append(model.names[cls_id])
        
        if detections:
            print(f"Processed {os.path.basename(img_path)}: {', '.join(detections)}")
        else:
            print(f"Processed {os.path.basename(img_path)}: No detections")

    print("\n--- Detection Summary ---")
    print(f"Total Images Processed: {total_images}")
    print(f"License Plates Detected: {summary[0]}")
    print(f"No License Plates Detected: {summary[1]}")

if __name__ == "__main__":
    run_inference()
