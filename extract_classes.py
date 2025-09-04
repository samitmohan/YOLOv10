#!/usr/bin/env python3
import json
import os
from collections import Counter


def extract_classes_from_json_files(json_root_dir):
    # Returns: tuple: (unique_classes_list, class_counts_dict)
    class_counts = Counter()
    processed_files = 0
    
    print(f"Scanning JSON files in: {json_root_dir}")
    
    # Walk through all subdirectories to find JSON files
    for root, dirs, files in os.walk(json_root_dir):
        for file in files:
            if file.lower().endswith('.json'):
                json_file_path = os.path.join(root, file)
                try:
                    with open(json_file_path, 'r') as f:
                        data = json.load(f)
                    
                    shapes = data.get('shapes', [])
                    for shape in shapes:
                        if shape.get('shape_type') == 'rectangle':
                            label = shape.get('label', '').strip()
                            if label:
                                # Normalize the label (lowercase for consistency)
                                normalized_label = label.lower()
                                class_counts[normalized_label] += 1
                    
                    processed_files += 1
                    if processed_files % 100 == 0:
                        print(f"  Processed {processed_files} JSON files...")
                        
                except Exception as e:
                    print(f"Error reading {json_file_path}: {e}")
    
    print(f"\nProcessed {processed_files} JSON files")
    
    unique_classes = sorted(class_counts.keys(), key=lambda x: (-class_counts[x], x))
    
    return unique_classes, dict(class_counts)


def display_class_statistics(unique_classes, class_counts):
    print(f"Total unique classes: {len(unique_classes)}")
    print(f"Total annotations: {sum(class_counts.values())}")
    print(f"{'Rank':<4} {'Class':<20} {'Count':<8} {'Percentage':<10}")
    
    total_annotations = sum(class_counts.values())
    for i, class_name in enumerate(unique_classes, 1):
        count = class_counts[class_name]
        percentage = (count / total_annotations) * 100
        print(f"{i:<4} {class_name:<20} {count:<8} {percentage:<10.2f}%")


def save_classes_to_file(unique_classes, class_counts, output_file):
    with open(output_file, 'w') as f:
        f.write("# Extracted Classes from JSON Files\n")
        f.write(f"# Total unique classes: {len(unique_classes)}\n")
        f.write(f"# Total annotations: {sum(class_counts.values())}\n\n")
        
        f.write("# Classes for YOLO dataset.yaml (sorted by frequency):\n")
        f.write("classes:\n")
        for class_name in unique_classes:
            f.write(f"  - {class_name}\n")
        
        total_annotations = sum(class_counts.values())
        for i, class_name in enumerate(unique_classes, 1):
            count = class_counts[class_name]
            percentage = (count / total_annotations) * 100
            f.write(f"# {i:2d}. {class_name:<20} {count:>6} ({percentage:5.2f}%)\n")
    
    print(f"\nClass information saved to: {output_file}")


def main():
    json_root_dir = "/home/samit/samit_workspace/training/odn/data/json" 
    output_file = "/home/samit/samit_workspace/training/odn/extracted_classes_2.txt"
    
    print(f"JSON root directory: {json_root_dir}")
    
    if not os.path.exists(json_root_dir):
        print(f"Error: JSON root directory not found: {json_root_dir}")
        return
    
    # Extract classes
    unique_classes, class_counts = extract_classes_from_json_files(json_root_dir)
    
    display_class_statistics(unique_classes, class_counts)
    
    save_classes_to_file(unique_classes, class_counts, output_file)
    
    print("names:")
    for class_name in unique_classes:
        print(f"  - {class_name}")


if __name__ == "__main__":
    main()