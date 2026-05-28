from ultralytics import YOLO
import torch
import os

def train():
    # Detect device
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = YOLO('yolov8n.pt') 
    
    # Dataset path
    data_yaml = os.path.abspath('data.yaml')
    
    # Training
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=16,           # Optimized for yolov8n
        workers=0,          # Safer for Windows multiprocessing
        device=device,
        patience=20,         # Early stopping
        save=True,
        cache=False,        # Disable caching if RAM is an issue
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,        # Tilted plates
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0001,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,           # Advanced augmentation
        copy_paste=0.0,
        auto_augment='randaugment',
        erasing=0.4,
        crop_fraction=1.0,
        cfg=None,
        tracker='botsort.yaml',
        # Optimization for small objects
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True,
        amp=True,            # Mixed precision
        cos_lr=True,         # Cosine learning rate
        close_mosaic=10      # Disable mosaic for last 10 epochs
    )
    
    print("Training completed.")
    return results

if __name__ == "__main__":
    train()
