import os
import shutil
from tqdm import tqdm

def merge_datasets(base_path, output_path):
    splits = {
        'train': [('with_plate/train', 'with'), ('without_plate/train', 'without')],
        'val': [('with_plate/valid', 'with'), ('without_plate/val', 'without')],
        'test': [('with_plate/test', 'with'), ('without_plate/test', 'without')]
    }
    
    for split, sources in splits.items():
        print(f"Processing {split} split...")
        img_out = os.path.join(output_path, split, 'images')
        lbl_out = os.path.join(output_path, split, 'labels')
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        
        for src_rel, prefix in sources:
            src_path = os.path.join(base_path, src_rel)
            if not os.path.exists(src_path):
                print(f"[WARNING] Source path {src_path} does not exist. Skipping.")
                continue
                
            img_src = os.path.join(src_path, 'images')
            lbl_src = os.path.join(src_path, 'labels')
            
            if not os.path.exists(img_src):
                print(f"[WARNING] Images path {img_src} does not exist.")
                continue

            for file in tqdm(os.listdir(img_src), desc=f"Copying {src_rel} images"):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    # Use prefix to avoid name collisions
                    new_name = f"{prefix}_{file}"
                    shutil.copy2(os.path.join(img_src, file), os.path.join(img_out, new_name))
                    
                    # Corresponding label
                    label_file = os.path.splitext(file)[0] + '.txt'
                    src_label_path = os.path.join(lbl_src, label_file)
                    dst_label_path = os.path.join(lbl_out, os.path.splitext(new_name)[0] + '.txt')
                    
                    if os.path.exists(src_label_path):
                        shutil.copy2(src_label_path, dst_label_path)
                    else:
                        # Create empty label file for background images
                        with open(dst_label_path, 'w') as f:
                            pass

def create_yaml(output_path):
    yaml_content = f"""
path: {os.path.abspath(output_path)}
train: train/images
val: val/images
test: test/images

names:
  0: license_plate
  1: no_license_plate
"""
    with open(os.path.join(os.path.dirname(output_path), 'data.yaml'), 'w') as f:
        f.write(yaml_content.strip())
    print("data.yaml created.")

if __name__ == "__main__":
    BASE = r"D:\License-Plate-Detection-main\Normalized_dataset"
    OUTPUT = r"D:\License-Plate-Detection-main\YOLO_Dataset"
    merge_datasets(BASE, OUTPUT)
    create_yaml(OUTPUT)
