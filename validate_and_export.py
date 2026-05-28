from ultralytics import YOLO
import os

def validate_and_export():
    # Use the best model from the latest run
    # Assuming the latest run is in runs/detect/train
    model_path = 'runs/detect/train/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model = YOLO(model_path)
    
    print("Running validation on test set...")
    metrics = model.val(data='data.yaml', split='test')
    
    print(f"mAP50: {metrics.results_dict['metrics/mAP50(B)']}")
    print(f"mAP50-95: {metrics.results_dict['metrics/mAP50-95(B)']}")
    print(f"Precision: {metrics.results_dict['metrics/precision(B)']}")
    print(f"Recall: {metrics.results_dict['metrics/recall(B)']}")
    
    print("Exporting model to ONNX...")
    model.export(format='onnx', imgsz=640, dynamic=True)
    
    print("Exporting model to TorchScript...")
    model.export(format='torchscript', imgsz=640)
    
    print("Export completed.")

if __name__ == "__main__":
    validate_and_export()
