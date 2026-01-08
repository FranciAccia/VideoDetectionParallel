import os
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO

INPUT_FOLDER = "videos"
OUTPUT_FOLDER = "output_videos_m2"
MODEL_NAME = "yolov8n.pt"
IMG_SIZE = 640


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def warmup_model(model, device, imgsz):
    dummy_input = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    model(dummy_input, device=device, verbose=False, imgsz=imgsz)


def process_video(video_path, model, device):
    filename = os.path.basename(video_path)
    out_path = os.path.join(OUTPUT_FOLDER, "det_" + filename)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening {filename}")
        return 0, 0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    count = 0
    start_t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, device=device, imgsz=IMG_SIZE, verbose=False)
        res_plotted = results[0].plot()

        writer.write(res_plotted)
        count += 1

    cap.release()
    writer.release()

    duration = time.time() - start_t
    return count, duration


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    device = get_device()
    print(f"Device: {device.upper()}")

    model = YOLO(MODEL_NAME)

    # Warmup per inizializzare i grafi di calcolo
    warmup_model(model, device, IMG_SIZE)

    video_files = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER)
                   if f.endswith(('.mp4', '.avi', '.mov'))]

    if not video_files:
        print("No video files found.")
        exit()

    print(f"Processing {len(video_files)} videos...")

    total_start = time.time()
    total_frames_all = 0

    for v in video_files:
        n_frames, t_video = process_video(v, model, device)
        fps = n_frames / t_video if t_video > 0 else 0
        total_frames_all += n_frames
        print(f"Processed {os.path.basename(v)}: {t_video:.2f}s ({fps:.1f} FPS)")

    total_duration = time.time() - total_start
    avg_fps = total_frames_all / total_duration if total_duration > 0 else 0

    print("-" * 30)
    print("RESULTS")
    print("-" * 30)
    print(f"Total Time:    {total_duration:.2f} s")
    print(f"Total Frames:  {total_frames_all}")
    print(f"Average FPS:   {avg_fps:.2f}")
    print("-" * 30)