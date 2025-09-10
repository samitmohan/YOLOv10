#!/usr/bin/python3

from ultralytics import YOLO
import torch
import os
import yaml
from collections import defaultdict
import numpy as np

def calculate_class_weights(data_yaml_path='data.yaml'):
    ''' For class imbalance '''
    with open(data_yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)

    dataset_root = data_yaml.get('path', '')
    names = data_yaml.get('names', [])
    num_classes = len(names)

    train_label_dir = os.path.join(dataset_root, data_yaml.get('train', '').replace('images', 'labels'))

    class_counts = defaultdict(int)

    for filename in os.listdir(train_label_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(train_label_dir, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    try:
                        class_id = int(line.strip().split(' ')[0])
                        class_counts[class_id] += 1
                    except (ValueError, IndexError):
                        pass
    
    # Calculate inverse frequency weights
    weights = torch.ones(num_classes)
    total_instances = sum(class_counts.values())
    
    if total_instances == 0:
        print("No instances found in training data, returning equal weights.")
        return weights

    for i in range(num_classes):
        count = class_counts.get(i, 0)
        if count > 0:
            # Using inverse frequency: 1 / frequency
            weights[i] = 1.0 / (count / total_instances)
        else:
            # Assign a very high weight to classes with no instances
            weights[i] = total_instances * 10 # or some other large number

    weights = weights / weights.mean()
    
    print("Calculated Class Weights:", weights.tolist())
    return weights

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint_path = 'runs/detect/resume_class_weighted_v10m_final/weights/best.pt'
    #use default/custom model (try with yolov10l.pt)
    model = YOLO(checkpoint_path)
    
    training_config = {
        'data': 'data.yaml',
        'epochs': 80,           
        'imgsz': 640,           
        'batch': 8,             
        'device': device,
        
        # Aggressive loss weights for class balance
        'box': 12.0,            # High box loss weight
        'cls': 3.0,             # High classification weight
        'dfl': 2.0,
        
        'optimizer': 'AdamW',
        'lr0': 0.0005,          # lower LR
        'lrf': 0.00001,
        'momentum': 0.9,
        'weight_decay': 0.0002,
        'cos_lr': True,
        
        'patience': 25,
        'warmup_epochs': 3,     
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.05,
        'close_mosaic': 30,
        
        # augmentation for CCTV
        'hsv_h': 0.01,
        'hsv_s': 0.3,
        'hsv_v': 0.2,
        'degrees': 2.0,
        'translate': 0.05,
        'scale': 0.8,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.3,
        'bgr': 0.0,
        
        # Augmentation for rare classes
        'mosaic': 0.4,
        'mixup': 0.0,
        'cutmix': 0.0,
        'copy_paste': 0.6,     
        'copy_paste_mode': 'flip',
        'auto_augment': False,
        'erasing': 0.1,
        
        'cache': 'disk',        
        'workers': 4,         
        'save_period': 10,      
        'amp': True,         
        
        'val': True,
        'plots': True,
        'save': True,
        'verbose': True,
        'save_json': True,
        
        'project': 'runs/detect',
        'name': 'resume_class_weighted_v10m_final',
        'exist_ok': False,
        'cls_weights': calculate_class_weights('data.yaml') # Add class weights
    }
    
    print(f"Configuration: batch={training_config['batch']}, imgsz={training_config['imgsz']}")
    print(f"Loss weights: box={training_config['box']}, cls={training_config['cls']}")
    print(f"Mosaic closes at epoch {training_config['close_mosaic']} for stabilization")
    
    results = model.train(**training_config)

    print("Running validation")
    best_model = YOLO('runs/detect/resume_class_weighted_v10m_final/weights/best.pt')
    val_results = best_model.val(data='data.yaml', conf=0.001, iou=0.6)
    
    print(f"mAP50: {val_results.box.map50:.3f}")
    print(f"mAP50-95: {val_results.box.map:.3f}")
    
if __name__ == "__main__":
    main()