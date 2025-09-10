#!/usr/bin/env python3
import os
import shutil
import random

def create_train_val_test_split(merged_data_dir, output_dir, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Creates train and validation splits from merged dataset for YOLO training.
    Current images: ~30k
    70% training, 15% validation, and 15% testing
    Output: real_dataset/train/images, real_dataset/train/labels, real_dataset/val/images, real_dataset/val/labels, real_dataset/test/images, real_dataset/test/labels
    """
    
    random.seed(seed)
    
    merged_images_dir = os.path.join(merged_data_dir, 'images')
    merged_labels_dir = os.path.join(merged_data_dir, 'labels')

    # output directory structure
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    val_labels_dir = os.path.join(output_dir, 'val', 'labels')
    test_images_dir = os.path.join(output_dir, 'test', 'images')
    test_labels_dir = os.path.join(output_dir, 'test', 'labels')
    
    for directory in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, test_images_dir, test_labels_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Get all image files
    image_files = []
    for file in os.listdir(merged_images_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            # corresponding label exist
            label_file = os.path.splitext(file)[0] + '.txt'
            label_path = os.path.join(merged_labels_dir, label_file)
            
            if os.path.exists(label_path):
                image_files.append(file)
            else:
                print(f"Warning: No label file found for {file}")
    
    # shuffle the files
    random.shuffle(image_files)
    
    # calculate split
    total_files = len(image_files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count
    
    print(f"Total files: {total_files}")
    print(f"Training: {train_count} ({train_count/total_files*100:.1f}%)")
    print(f"Validation: {val_count} ({val_count/total_files*100:.1f}%)")
    print(f"Test: {test_count} ({test_count/total_files*100:.1f}%)")
    
    # Split files
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]
    
    # Copy training files
    print("Copying training files")
    for i, img_file in enumerate(train_files):
        if i % 1000 == 0:
            print(f"  Progress: {i}/{len(train_files)}")
        
        # Copy image
        src_img = os.path.join(merged_images_dir, img_file)
        dst_img = os.path.join(train_images_dir, img_file)
        shutil.copy2(src_img, dst_img)
        
        # Copy label
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label = os.path.join(merged_labels_dir, label_file)
        dst_label = os.path.join(train_labels_dir, label_file)
        shutil.copy2(src_label, dst_label)
    
    # Copy validation files
    print("Copying validation files")
    for i, img_file in enumerate(val_files):
        if i % 1000 == 0:
            print(f"  Progress: {i}/{len(val_files)}")
        
        # Copy image
        src_img = os.path.join(merged_images_dir, img_file)
        dst_img = os.path.join(val_images_dir, img_file)
        shutil.copy2(src_img, dst_img)
        
        # Copy label
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label = os.path.join(merged_labels_dir, label_file)
        dst_label = os.path.join(val_labels_dir, label_file)
        shutil.copy2(src_label, dst_label)
    
    # Copy test files
    print("Copying test files")
    for i, img_file in enumerate(test_files):
        if i % 1000 == 0:
            print(f"  Progress: {i}/{len(test_files)}")
        
        # Copy image
        src_img = os.path.join(merged_images_dir, img_file)
        dst_img = os.path.join(test_images_dir, img_file)
        shutil.copy2(src_img, dst_img)
        
        # Copy label
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label = os.path.join(merged_labels_dir, label_file)
        dst_label = os.path.join(test_labels_dir, label_file)
        shutil.copy2(src_label, dst_label)
    
    # output structure
    print(f"  {output_dir}/")
    print("    train/")
    print(f"      images/ ({len(train_files)} files)")
    print(f"      labels/ ({len(train_files)} files)")
    print("    val/")
    print(f"      images/ ({len(val_files)} files)")
    print(f"      labels/ ({len(val_files)} files)")
    print("    test/")
    print(f"      images/ ({len(test_files)} files)")
    print(f"      labels/ ({len(test_files)} files)")


def main():
    merged_data_dir = "/home/samit/samit_workspace/training/odn/merged_data"
    output_dir = "/home/samit/samit_workspace/training/odn/dataset"
    
    print(f"Input directory: {merged_data_dir}")
    print(f"Output directory: {output_dir}")
    
    if not os.path.exists(merged_data_dir):
        print(f"Merged data directory not found: {merged_data_dir}")
        return
    
    create_train_val_test_split(merged_data_dir, output_dir, train_ratio=0.70, val_ratio=0.15)


if __name__ == "__main__":
    main()