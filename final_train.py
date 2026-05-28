from ultralytics import YOLO
import torch
import os
import sys

def check_env():
    print("Checking Environment...")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    return device

def train():
    device = check_env()
    
    # Load YOLOv8n model
    model = YOLO('yolov8n.pt')
    
    # Dataset YAML path
    data_yaml = os.path.abspath('dataset.yaml')
    
    # Start Training
    # Requirements: 100 epochs, imgsz 640, auto-batch, early stopping, augmentation
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=-1,           # Auto-batch size
        device=device,
        workers=0,          # Safer for Windows
        patience=20,         # Early stopping
        save=True,
        cache=False,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        augment=True,        # Enable default augmentations
        mosaic=1.0,          # Strong mosaic
        mixup=0.1,           # Mixup for complex scenes
        copy_paste=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        name='license_plate_dual_class'
    )
    
    print("Training Complete.")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")
    
    # Export model as requested
    print("Exporting model...")
    model.export(format='onnx')
    model.export(format='torchscript')

if __name__ == "__main__":
    train()
