# Model evaluation on unseen data (Test : 1.5k images for now)
import time
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def benchmark(model_path, test_dir):
    model = YOLO(model_path)
    images = list(Path(test_dir).glob('*.jpg'))[:50]

    times = []
    for img in images:
        start = time.perf_counter()
        model(str(img), verbose=False)
        times.append((time.perf_counter() - start) * 1000)

    return np.mean(times)

# Compare
v8_time = benchmark('/home/cv-laptop-1/Desktop/samit/triton/objectdetection/runs/detect/train_v8m/weights/best.pt', 'dataset/test/images')
v10_time = benchmark('/home/cv-laptop-1/Desktop/samit/triton/objectdetection/runs/detect/train_v10m/weights/best.pt', 'dataset/test/images')

print(f"YOLOv8: {v8_time:.2f}ms")
print(f"YOLOv10: {v10_time:.2f}ms")
print(f"Speed improvement: {((v8_time-v10_time)/v8_time)*100:.1f}%")
