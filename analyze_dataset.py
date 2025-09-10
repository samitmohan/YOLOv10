import os
import yaml
from collections import defaultdict
import cv2
import numpy as np

def count_class_instances(label_dir, names):
    class_counts = defaultdict(int)

    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(label_dir, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split(' ')[0])
                    class_counts[class_id] += 1
    return class_counts

def draw_bounding_boxes(image_dir, label_dir, output_dir, class_names):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # BGR colors for car, truck, man, bike, bus, bicycle, auto
    colors = [
        (0, 255, 255),  # car (yellow)
        (0, 0, 255),    # truck (red)
        (0, 0, 0),      # man (black)
        (0, 255, 0),    # bike (green)
        (255, 0, 0),    # bus (blue)
        (255, 0, 255),  # bicycle (magenta)
        (255, 255, 0)   # auto (cyan)
    ]

    for image_filename in os.listdir(image_dir):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, image_filename)
            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            label_path = os.path.join(label_dir, label_filename)

            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                continue
            
            h, w, _ = image.shape

            if not os.path.exists(label_path):
                print(f"Warning: Label file not found for image {image_filename}")
                continue

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])

                    x_min = int((x_center - width / 2) * w)
                    y_min = int((y_center - height / 2) * h)
                    x_max = int((x_center + width / 2) * w)
                    y_max = int((y_center + height / 2) * h)

                    color = colors[class_id]
                    class_name = class_names[class_id]

                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            output_path = os.path.join(output_dir, image_filename)
            cv2.imwrite(output_path, image)
    print(f"Processed images with bounding boxes are saved in: {output_dir}")


def analyze_dataset(data_yaml_path='data.yaml'):
    with open(data_yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)

    dataset_root = data_yaml.get('path', '')
    names = data_yaml.get('names', [])

    splits = {
        'train': data_yaml.get('train', '').replace('images', 'labels'),
        'val': data_yaml.get('val', '').replace('images', 'labels'),
        'test': data_yaml.get('test', '').replace('images', 'labels')
    }

    print(f"Analyzing dataset from: {dataset_root}")
    print("Class Names:", names)

    for split_name, relative_label_path in splits.items():
        full_label_dir = os.path.join(dataset_root, relative_label_path)
        print(f"Class distribution for {split_name} split ({full_label_dir}):")
        class_counts = count_class_instances(full_label_dir, names)

        if not class_counts:
            print("No instances found")
        else:
            for class_id in sorted(class_counts.keys()):
                if class_id < len(names):
                    class_name = names[class_id]
                else:
                    class_name = f'Unknown_{class_id}'
                print(f"  {class_name}: {class_counts[class_id]} instances")

def visualize_test_set(data_yaml_path='data.yaml'):
    with open(data_yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
    
    dataset_root = data_yaml.get('path', '')
    names = data_yaml.get('names', [])
    test_images_rel_path = data_yaml.get('test', '')
    test_labels_rel_path = test_images_rel_path.replace('images', 'labels')

    image_dir = os.path.join(dataset_root, test_images_rel_path)
    label_dir = os.path.join(dataset_root, test_labels_rel_path)
    output_dir = 'test_dataset_with_bboxes'

    print(f"\nDrawing bounding boxes for test set...")
    draw_bounding_boxes(image_dir, label_dir, output_dir, names)


if __name__ == '__main__':
    analyze_dataset()
    visualize_test_set()
