import subprocess
import sys
import os

def check_dependencies():
    required = [
        'ultralytics', 'torch', 'torchvision', 'torchaudio', 
        'opencv-python', 'numpy', 'matplotlib', 'pandas', 
        'Pillow', 'tqdm', 'scikit-learn', 'tensorboard', 'onnx'
    ]
    
    print("Checking dependencies...")
    for lib in required:
        try:
            __import__(lib.replace('-', '_'))
            print(f"[OK] {lib}")
        except ImportError:
            print(f"[MISSING] {lib}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

def check_cuda():
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            print("[WARNING] CUDA not available. Attempting to install torch with CUDA support...")
            # For RTX 3050, we want CUDA 11.8 or 12.1
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])
    except ImportError:
        print("[ERROR] torch not installed.")

if __name__ == "__main__":
    check_dependencies()
    check_cuda()
