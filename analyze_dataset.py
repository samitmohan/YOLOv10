import os
import yaml
from collections import defaultdict

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

if __name__ == '__main__':
    analyze_dataset()
