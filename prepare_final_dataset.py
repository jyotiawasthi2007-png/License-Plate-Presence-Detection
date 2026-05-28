import os
import shutil
from tqdm import tqdm

def fix_and_merge_datasets():
    # Paths
    dataset_raw = r"D:\License-Plate-Detection-main\Dataset"
    norm_dataset = r"D:\License-Plate-Detection-main\Normalized_dataset"
    output_path = r"D:\License-Plate-Detection-main\Final_YOLO_Dataset"
    
    splits = {
        'train': [('with_plate/train', 'without_plate/train')],
        'valid': [('with_plate/valid', 'without_plate/val')], 
        'test': [('with_plate/test', 'without_plate/test')]
    }
    
    # Label source for 'without' from Dataset (raw)
    without_label_source = os.path.join(dataset_raw, 'without licence plate', 'label')

    # Load all 'without' labels into memory
    without_labels = {}
    if os.path.exists(without_label_source):
        for f in os.listdir(without_label_source):
            if f.endswith('.txt'):
                with open(os.path.join(without_label_source, f), 'r') as file:
                    without_labels[os.path.splitext(f)[0]] = file.readlines()

    for split, source_pairs in splits.items():
        print(f"Processing {split} split...")
        img_out = os.path.join(output_path, split, 'images')
        lbl_out = os.path.join(output_path, split, 'labels')
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        
        with_src, without_src = source_pairs[0]
        
        # Process 'with_plate' - Copy directly from Normalized_dataset
        src_with_img = os.path.join(norm_dataset, with_src, 'images')
        src_with_lbl = os.path.join(norm_dataset, with_src, 'labels')
        if os.path.exists(src_with_img):
            for f in tqdm(os.listdir(src_with_img), desc=f"Copying with_plate {split}"):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    shutil.copy2(os.path.join(src_with_img, f), os.path.join(img_out, f))
                    lbl_f = os.path.splitext(f)[0] + '.txt'
                    if os.path.exists(os.path.join(src_with_lbl, lbl_f)):
                        shutil.copy2(os.path.join(src_with_lbl, lbl_f), os.path.join(lbl_out, lbl_f))

        # Process 'without_plate' - Map from Dataset/without...
        src_without_img = os.path.join(norm_dataset, without_src, 'images')
        if os.path.exists(src_without_img):
            for f in tqdm(os.listdir(src_without_img), desc=f"Mapping without_plate {split}"):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    shutil.copy2(os.path.join(src_without_img, f), os.path.join(img_out, f))
                    
                    base_name = os.path.splitext(f)[0]
                    lookup_name = base_name.split('_aug_')[0] if '_aug_' in base_name else base_name
                    
                    lines = without_labels.get(lookup_name, [])
                    dst_lbl_path = os.path.join(lbl_out, base_name + '.txt')
                    with open(dst_lbl_path, 'w') as out_f:
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                parts[0] = '1' # Force class 1
                                out_f.write(" ".join(parts) + "\n")

def create_final_yaml():
    yaml_content = f"""
path: {os.path.abspath('Final_YOLO_Dataset')}
train: train/images
val: valid/images
test: test/images

names:
  0: license_plate
  1: no_license_plate
"""
    with open('dataset.yaml', 'w') as f:
        f.write(yaml_content.strip())
    print("dataset.yaml created.")

if __name__ == "__main__":
    if os.path.exists('Final_YOLO_Dataset'):
        shutil.rmtree('Final_YOLO_Dataset')
    fix_and_merge_datasets()
    create_final_yaml()
