#!/usr/bin/python3

import os
import glob

def clean_jcb_labels():
    """Remove JCB class (class 5) from all label files and adjust class indices"""
    
    dataset_path = "/home/cv-laptop-1/Desktop/samit/triton/objectdetection/real_dataset"
    label_dirs = [
        os.path.join(dataset_path, "train", "labels"),
        os.path.join(dataset_path, "val", "labels"),
        os.path.join(dataset_path, "test", "labels")
    ]
    
    total_files = 0
    modified_files = 0
    removed_annotations = 0
    
    # Class mapping: old_class -> new_class (removing JCB=5, adjusting bicycle=6->5, auto=7->6)
    class_mapping = {
        0: 0,  # car -> car
        1: 1,  # truck -> truck  
        2: 2,  # man -> man
        3: 3,  # bike -> bike
        4: 4,  # bus -> bus
        5: None,  # jcb -> remove
        6: 5,  # bicycle -> bicycle
        7: 6   # auto -> auto
    }
    
    for label_dir in label_dirs:
        if not os.path.exists(label_dir):
            continue
            
        label_files = glob.glob(os.path.join(label_dir, "*.txt"))
        total_files += len(label_files)
        
        for file_path in label_files:
            lines = []
            modified = False
            
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            old_class = int(parts[0])
                            
                            if old_class == 5:  # Remove JCB
                                removed_annotations += 1
                                modified = True
                                continue
                            
                            new_class = class_mapping.get(old_class, old_class)
                            if new_class is not None:
                                parts[0] = str(new_class)
                                if new_class != old_class:
                                    modified = True
                                lines.append(' '.join(parts) + '\n')
                
                if modified:
                    with open(file_path, 'w') as f:
                        f.writelines(lines)
                    modified_files += 1
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    print(f"Processed {total_files} label files")
    print(f"Modified {modified_files} files") 
    print(f"Removed {removed_annotations} JCB annotations")
    print("Class indices updated: bicycle (6->5), auto (7->6)")

if __name__ == "__main__":
    clean_jcb_labels()