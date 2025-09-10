#!/usr/bin/python3

from ultralytics import YOLO
import cv2
import os
import numpy as np
from pathlib import Path
import torch

def analyze_false_positives():
    """Run inference on validation set and analyze false positives"""
    
    model = YOLO('/home/cv-laptop-1/Desktop/samit/triton/objectdetection/runs/detect/resume_class_weighted_v10m_final/weights/best.pt')
    
    val_images_dir = "/home/cv-laptop-1/Desktop/samit/triton/objectdetection/real_dataset/val/images"
    val_labels_dir = "/home/cv-laptop-1/Desktop/samit/triton/objectdetection/real_dataset/val/labels"
    
    # Get sample images for analysis
    import glob
    val_images = glob.glob(os.path.join(val_images_dir, "*.jpg"))[:20]  # Analyze 20 images
    
    class_names = ['car', 'truck', 'man', 'bike', 'bus', 'bicycle', 'auto']
    
    print(f"Analyzing {len(val_images)} validation images...")
    
    false_positives = {}
    confidence_distribution = {}
    
    for class_name in class_names:
        false_positives[class_name] = []
        confidence_distribution[class_name] = []
    
    for i, img_path in enumerate(val_images[:10]):  # First 10 for detailed analysis
        print(f"Processing image {i+1}: {os.path.basename(img_path)}")
        
        # Load ground truth labels
        label_path = os.path.join(val_labels_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
        
        ground_truth = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, x, y, w, h = map(float, parts[:5])
                        ground_truth.append([int(cls), x, y, w, h])
        
        # Run inference
        results = model(img_path, conf=0.25, iou=0.45, verbose=False)
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                if cls_id < len(class_names):
                    class_name = class_names[cls_id]
                    confidence_distribution[class_name].append(conf)
                    
                    # Check if this is a potential false positive
                    if conf < 0.5:
                        false_positives[class_name].append({
                            'image': os.path.basename(img_path),
                            'confidence': conf,
                            'bbox': box.xywhn[0].tolist() if len(box.xywhn) > 0 else []
                        })
    
    print("FALSE POSITIVE ANALYSIS")
    for class_name in class_names:
        fps = false_positives[class_name]
        confs = confidence_distribution[class_name]
        
        if confs:
            avg_conf = np.mean(confs)
            low_conf_count = sum(1 for c in confs if c < 0.5)
            print(f"\n{class_name.upper()}:")
            print(f"  Total detections: {len(confs)}")
            print(f"  Average confidence: {avg_conf:.3f}")
            print(f"  Low confidence (<0.5): {low_conf_count} ({low_conf_count/len(confs)*100:.1f}%)")
            print(f"  Potential false positives: {len(fps)}")
    
    # Run validation to get detailed metrics
    print("RUNNING VALIDATION")
    val_results = model.val(data='data.yaml', conf=0.001, iou=0.6, verbose=True)
    
    print(f"Current mAP50: {val_results.box.map50:.3f}")
    print(f"Current mAP50-95: {val_results.box.map:.3f}")
    
    # Per-class analysis
    if hasattr(val_results.box, 'maps'):
        print("\nPer-class mAP50-95:")
        for i, class_name in enumerate(class_names):
            if i < len(val_results.box.maps):
                print(f"  {class_name}: {val_results.box.maps[i]:.3f}")

if __name__ == "__main__":
    analyze_false_positives()