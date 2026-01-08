"""
Video Object Detection - Sequential vs Parallel
For Parallel Programming Exam - MacBook Air M2
"""

import cv2
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Global variable for model name (needed for worker processes)
MODEL_NAME = 'yolov8n.pt'


def process_frame_batch(args):
    """
    Worker function to process a batch of frames.
    Each worker loads its own YOLO model. 
    """
    frames_data, indices, model_name = args
    
    # Import YOLO inside worker to avoid multiprocessing issues
    from ultralytics import YOLO
    model = YOLO(model_name)
    
    detections = 0
    annotated_frames = {}
    
    for frame, idx in zip(frames_data, indices):
        results = model(frame, verbose=False)
        for result in results: 
            detections += len(result. boxes)
        annotated_frames[idx] = results[0]. plot()
    
    return {'detections': detections, 'annotated_frames': annotated_frames}


def get_video_files(videos_folder='videos'):
    """Get all video files from the folder"""
    videos_path = Path(videos_folder)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.MP4', '.MOV', '.AVI']
    video_files = []
    
    if not videos_path.exists():
        videos_path.mkdir(exist_ok=True)
        return []
    
    for ext in video_extensions:
        video_files.extend(videos_path.glob(f'*{ext}'))
    
    return sorted(list(video_files))


def process_video_sequential(video_path, model, save_output=False, output_folder='output'):
    """Process a single video sequentially"""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap. get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = None
    if save_output: 
        output_path = Path(output_folder) / f"seq_{video_path. name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    start_time = time. time()
    
    print(f"\n  Processing: {video_path.name}")
    print(f"  Frames: {total_frames}, FPS: {fps}, Resolution: {width}x{height}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, verbose=False)
        
        for result in results: 
            total_detections += len(result. boxes)
        
        annotated_frame = results[0].plot()
        
        if save_output and out is not None: 
            out.write(annotated_frame)
        
        frame_count += 1
        
        if frame_count % 50 == 0:
            elapsed = time.time() - start_time
            fps_proc = frame_count / elapsed if elapsed > 0 else 0
            print(f"    Frame {frame_count}/{total_frames} ({fps_proc:.1f} FPS)")
    
    elapsed_time = time.time() - start_time
    
    cap.release()
    if out is not None: 
        out.release()
    
    print(f"  ✓ Completed in {elapsed_time:.2f}s - {frame_count/elapsed_time:.2f} FPS")
    
    return {
        'video_name': video_path. name,
        'frames_processed': frame_count,
        'total_detections': total_detections,
        'elapsed_time': elapsed_time,
        'fps_processing': frame_count / elapsed_time if elapsed_time > 0 else 0
    }


def process_video_parallel(video_path, model_name, num_processes, save_output=False, output_folder='output'):
    """Process a single video with parallel frame processing"""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None
    
    fps = int(cap. get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n  Processing: {video_path.name}")
    print(f"  Frames: {total_frames}, FPS: {fps}, Resolution: {width}x{height}")
    print(f"  Using {num_processes} parallel processes")
    
    # Read all frames into memory
    print("  Reading frames into memory...")
    frames = []
    frame_indices = []
    idx = 0
    
    while True: 
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_indices.append(idx)
        idx += 1
    cap.release()
    
    if not frames:
        return None
    
    print(f"  ✓ Read {len(frames)} frames")
    
    # Split frames into batches for each process
    batch_size = max(1, len(frames) // num_processes)
    frame_batches = []
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        batch_indices = frame_indices[i:i + batch_size]
        frame_batches.append((batch_frames, batch_indices, model_name))
    
    print(f"  Created {len(frame_batches)} batches for parallel processing")
    print("  Starting parallel inference...")
    
    start_time = time. time()
    
    # Process batches in parallel
    with Pool(processes=num_processes) as pool:
        batch_results = pool. map(process_frame_batch, frame_batches)
    
    elapsed_time = time. time() - start_time
    
    # Combine results
    total_detections = 0
    all_annotated_frames = {}
    
    for batch_result in batch_results: 
        total_detections += batch_result['detections']
        all_annotated_frames.update(batch_result['annotated_frames'])
    
    # Save output video if requested
    if save_output:
        output_path = Path(output_folder) / f"par_{video_path.name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for i in range(len(frames)):
            if i in all_annotated_frames: 
                out.write(all_annotated_frames[i])
        
        out. release()
    
    print(f"  ✓ Completed in {elapsed_time:.2f}s - {len(frames)/elapsed_time:.2f} FPS")
    
    return {
        'video_name':  video_path.name,
        'frames_processed': len(frames),
        'total_detections':  total_detections,
        'elapsed_time': elapsed_time,
        'fps_processing': len(frames) / elapsed_time if elapsed_time > 0 else 0
    }


def plot_results(seq_results, par_results, seq_time, par_time, speedup, num_processes, output_folder):
    """Create comparison plots"""
    fig, axes = plt. subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Video Object Detection:  Sequential vs Parallel\n(MacBook Air M2)',
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Total time
    ax1 = axes[0, 0]
    methods = ['Sequential', 'Parallel']
    times = [seq_time, par_time]
    colors = ['#ff6b6b', '#4ecdc4']
    bars = ax1.bar(methods, times, color=colors, edgecolor='black')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Total Processing Time')
    ax1.grid(axis='y', alpha=0.3)
    for bar, t in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'{t:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: SpeedUp
    ax2 = axes[0, 1]
    categories = ['Ideal\n(Theoretical)', 'Actual\nSpeedUp']
    values = [num_processes, speedup]
    bars = ax2.bar(categories, values, color=['#95e1d3', '#4ecdc4'], edgecolor='black')
    ax2.axhline(y=1, color='red', linestyle='--', label='Baseline (1x)')
    ax2.set_ylabel('SpeedUp Factor')
    ax2.set_title(f'SpeedUp:  {speedup:.2f}x (Efficiency: {(speedup/num_processes)*100:.1f}%)')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    for bar, v in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'{v:.2f}x', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Time per video
    ax3 = axes[1, 0]
    video_names = [r['video_name'][:12] for r in seq_results]
    seq_times_list = [r['elapsed_time'] for r in seq_results]
    par_times_list = [r['elapsed_time'] for r in par_results]
    
    x = np.arange(len(video_names))
    width = 0.35
    
    ax3.bar(x - width/2, seq_times_list, width, label='Sequential', color='#ff6b6b')
    ax3.bar(x + width/2, par_times_list, width, label='Parallel', color='#4ecdc4')
    ax3.set_xlabel('Video')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Processing Time per Video')
    ax3.set_xticks(x)
    ax3.set_xticklabels(video_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    total_frames = sum(r['frames_processed'] for r in seq_results)
    total_det = sum(r['total_detections'] for r in seq_results)
    efficiency = (speedup / num_processes) * 100
    
    summary = f"""
    ╔══════════════════════════════════════════╗
    ║         PERFORMANCE SUMMARY              ║
    ╠══════════════════════════════════════════╣
    ║  Videos processed:      {len(seq_results):>14}  ║
    ║  Total frames:         {total_frames: >14}  ║
    ║  Objects detected:     {total_det: >14}  ║
    ║  Processes used:       {num_processes:>14}  ║
    ╠══════════════════════════════════════════╣
    ║  Sequential time:      {seq_time:>12.2f}s  ║
    ║  Parallel time:        {par_time:>12.2f}s  ║
    ║  Time saved:           {seq_time-par_time:>12.2f}s  ║
    ╠══════════════════════════════════════════╣
    ║  SpeedUp:               {speedup: >13.2f}x  ║
    ║  Efficiency:           {efficiency:>12.1f}%  ║
    ╚══════════════════════════════════════════╝
    """
    
    ax4.text(0.5, 0.5, summary, transform=ax4.transAxes,
             fontsize=10, va='center', ha='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_path = Path(output_folder) / 'performance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")


def main():
    """Main function - called from run.py"""
    print("=" * 70)
    print("VIDEO OBJECT DETECTION - SEQUENTIAL vs PARALLEL")
    print("Parallel Programming Project - MacBook Air M2")
    print("=" * 70)
    
    # Configuration
    global MODEL_NAME
    MODEL_NAME = 'yolov8n.pt'
    VIDEOS_FOLDER = 'videos'
    OUTPUT_FOLDER = 'output'
    SAVE_OUTPUT = True
    NUM_PROCESSES = cpu_count()
    
    # Create output folder
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Model:  {MODEL_NAME}")
    print(f"  Videos folder:  {VIDEOS_FOLDER}")
    print(f"  Output folder: {OUTPUT_FOLDER}")
    print(f"  CPU cores: {NUM_PROCESSES}")
    
    # Check for videos
    video_files = get_video_files(VIDEOS_FOLDER)
    
    if not video_files:
        print(f"\n⚠️  No videos found in '{VIDEOS_FOLDER}' folder!")
        print("   Please add video files (. mp4, .mov, . avi) and run again.")
        return
    
    print(f"\n✓ Found {len(video_files)} video(s):")
    for vf in video_files:
        size_mb = vf.stat().st_size / (1024 * 1024)
        print(f"  - {vf.name} ({size_mb:.1f} MB)")
    
    # Load YOLO model for sequential processing
    print("\nLoading YOLO model...")
    from ultralytics import YOLO
    model = YOLO(MODEL_NAME)
    print("✓ Model loaded successfully!")
    
    # ========== SEQUENTIAL PROCESSING ==========
    print("\n" + "=" * 70)
    print("SEQUENTIAL PROCESSING")
    print("=" * 70)
    
    seq_start = time.time()
    seq_results = []
    
    for video_path in video_files: 
        stats = process_video_sequential(video_path, model, SAVE_OUTPUT, OUTPUT_FOLDER)
        if stats:
            seq_results.append(stats)
    
    seq_time = time.time() - seq_start
    print(f"\n>>> SEQUENTIAL TOTAL TIME:  {seq_time:. 2f}s <<<")
    
    # ========== PARALLEL PROCESSING ==========
    print("\n" + "=" * 70)
    print(f"PARALLEL PROCESSING ({NUM_PROCESSES} processes)")
    print("=" * 70)
    
    par_start = time.time()
    par_results = []
    
    for video_path in video_files: 
        stats = process_video_parallel(video_path, MODEL_NAME, NUM_PROCESSES, SAVE_OUTPUT, OUTPUT_FOLDER)
        if stats:
            par_results.append(stats)
    
    par_time = time.time() - par_start
    print(f"\n>>> PARALLEL TOTAL TIME: {par_time:.2f}s <<<")
    
    # ========== RESULTS ==========
    speedup = seq_time / par_time if par_time > 0 else 0
    efficiency = (speedup / NUM_PROCESSES) * 100
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Sequential time:  {seq_time:. 2f}s")
    print(f"  Parallel time:    {par_time:.2f}s")
    print(f"  Time saved:       {seq_time - par_time:.2f}s")
    print(f"\n  >>> SPEEDUP:    {speedup:.2f}x <<<")
    print(f"  >>> EFFICIENCY:  {efficiency:.1f}% <<<")
    print("=" * 70)
    
    # Create plots
    plot_results(seq_results, par_results, seq_time, par_time, speedup, NUM_PROCESSES, OUTPUT_FOLDER)
    
    print("\n✓ BENCHMARK COMPLETED SUCCESSFULLY!")