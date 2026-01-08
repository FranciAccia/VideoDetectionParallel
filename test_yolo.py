#!/usr/bin/env python3
"""Test fix for ultralytics import hang on macOS"""

import os
import sys

# CRITICAL: Set these environment variables BEFORE importing ultralytics
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os. environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Disable tokenizers parallelism (can cause issues)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# For PyTorch on macOS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import time

print("Step 1: Environment variables set")
print(f"  Python:  {sys.executable}")

print("\nStep 2: Importing ultralytics...")
start = time.time()

from ultralytics import YOLO

print(f"  ✓ Import completed in {time.time() - start:.2f}s")

print("\nStep 3: Loading YOLO model...")
start = time.time()

model = YOLO('yolov8n.pt')

print(f"  ✓ Model loaded in {time. time() - start:.2f}s")

print("\nStep 4: Testing inference on a dummy image...")
import numpy as np
dummy_image = np. zeros((640, 640, 3), dtype=np.uint8)
results = model(dummy_image, verbose=False)
print(f"  ✓ Inference works!  Detected {len(results[0].boxes)} objects")

print("\n✓ ALL TESTS PASSED!")