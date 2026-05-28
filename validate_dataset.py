import os
import glob

def validate_dataset():
    base_path = 'Final_YOLO_Dataset'
    splits = ['train', 'valid', 'test']
    classes = {0: 'license_plate', 1: 'no_license_plate'}
    
    print(f"--- Dataset Validation: {base_path} ---")
    
    stats = {}
    
    for split in splits:
        img_dir = os.path.join(base_path, split, 'images')
        lbl_dir = os.path.join(base_path, split, 'labels')
        
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"Error: Missing {split} directories.")
            continue
            
        images = glob.glob(os.path.join(img_dir, '*.*'))
        labels = glob.glob(os.path.join(lbl_dir, '*.txt'))
        
        print(f"\nSplit: {split}")
        print(f"  Images: {len(images)}")
        print(f"  Labels: {len(labels)}")
        
        # Count classes
        class_counts = {0: 0, 1: 0}
        for lbl_path in labels:
            with open(lbl_path, 'r') as f:
                for line in f:
                    try:
                        cls_id = int(line.split()[0])
                        if cls_id in class_counts:
                            class_counts[cls_id] += 1
                    except:
                        pass
        
        for cid, count in class_counts.items():
            print(f"  - Class {cid} ({classes[cid]}): {count} boxes")

    print("\nValidation Complete.")

if __name__ == "__main__":
    validate_dataset()
