import os
# Vogliamo che ogni processo usi 1 solo core, così possiamo scalarne N in parallelo.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import cv2
import time
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count, set_start_method
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import platform


class VideoObjectDetection:
    def __init__(self, model_name='yolov8n.pt', videos_folder='videos', output_folder='output'):
        self.model_name = model_name
        self.videos_folder = Path(videos_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)

        # Verifica architettura per report
        print(f"Sistema rilevato: {platform.system()} {platform.machine()}"
              if platform.system() == 'Darwin' and 'arm' in platform.machine()
              else f"Sistema rilevato: {platform.system()} {platform.machine()}")

        # Carica il modello una volta per scaricare i pesi se necessario
        print(f"Check modello {model_name}...")
        _ = YOLO(model_name)
        print("Modello pronto.")

    def get_video_files(self):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        video_files = []
        if not self.videos_folder.exists():
            print(f"Errore: La cartella '{self.videos_folder}' non esiste!")
            return video_files
        for ext in video_extensions:
            video_files.extend(self.videos_folder.glob(f'*{ext}'))
        return list(video_files)

    def process_video_sequential(self, video_path, save_output=False):
        model = YOLO(self.model_name)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = None
        if save_output:
            output_path = self.output_folder / f"seq_{video_path.name}"
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        total_detections = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Forza CPU per coerenza con il test parallelo
            results = model(frame, verbose=False, device='cpu')

            for result in results:
                total_detections += len(result.boxes)

            if save_output and out is not None:
                annotated_frame = results[0].plot()
                out.write(annotated_frame)

            frame_count += 1

        elapsed_time = time.time() - start_time
        cap.release()
        if out is not None:
            out.release()

        return {
            'video_name': video_path.name,
            'frames_processed': frame_count,
            'total_detections': total_detections,
            'elapsed_time': elapsed_time
        }

    def process_all_videos_sequential(self, save_output=False, verbose=True):
        video_files = self.get_video_files()
        if not video_files: return [], 0

        if verbose:
            print(f"   [Sequential] Elaborazione {len(video_files)} video...")

        start_time = time.time()
        results = []
        for video_path in video_files:
            stats = self.process_video_sequential(video_path, save_output)
            if stats:
                results.append(stats)
        total_time = time.time() - start_time

        return results, total_time

    @staticmethod
    def process_video_worker(args):
        video_path, model_name, output_folder, save_output = args

        import torch
        torch.set_num_threads(1)  # Blocca il parallelismo intra-op di PyTorch
        torch.set_num_interop_threads(1)  # Blocca il parallelismo inter-op
        cv2.setNumThreads(0)
        model = YOLO(model_name)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = None
        if save_output:
            output_path = Path(output_folder) / f"par_{Path(video_path).name}"
            # Usiamo 'mp4v' o 'avc1' a seconda di cosa funziona meglio sul tuo Mac
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        total_detections = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Inferenza su CPU (il limite thread imposto sopra la renderà più lenta per singolo frame,
            # ma permetterà ai processi paralleli di scalare)
            results = model(frame, verbose=False, device='cpu')

            for result in results:
                total_detections += len(result.boxes)

            if save_output and out is not None:
                annotated_frame = results[0].plot()
                out.write(annotated_frame)

            frame_count += 1

        elapsed_time = time.time() - start_time
        cap.release()
        if out is not None:
            out.release()

        return {
            'video_name': Path(video_path).name,
            'frames_processed': frame_count,
            'total_detections': total_detections,
            'elapsed_time': elapsed_time
        }

    def process_all_videos_parallel(self, save_output=False, num_processes=None, verbose=True):
        video_files = self.get_video_files()
        if not video_files: return [], 0

        if num_processes is None:
            num_processes = cpu_count()

        actual_processes = min(num_processes, len(video_files))

        if verbose:
            print(f"   [Parallel] Elaborazione {len(video_files)} video con {actual_processes} processi...")

        worker_args = [
            (video_path, self.model_name, self.output_folder, save_output)
            for video_path in video_files
        ]

        start_time = time.time()
        # Su macOS con 'spawn', gli oggetti passati ai worker devono essere picklable.
        # YOLO e OpenCV gestiscono questo meglio se reinizializzati nel worker (come fatto sopra).
        with Pool(processes=actual_processes) as pool:
            results = pool.map(self.process_video_worker, worker_args)
        total_time = time.time() - start_time

        results = [r for r in results if r is not None]
        return results, total_time

    def run_robust_benchmark(self, save_output=False, n_runs=3):
        print("\n" + "=" * 80)
        print(f"AVVIO BENCHMARK (MacBook M2 Optimized)")
        print(f"Runs per configurazione: {n_runs}")
        print("=" * 80)

        video_files = self.get_video_files()
        if not video_files:
            print("Nessun video trovato nella cartella 'videos'!")
            return

        # 1. Warm-up
        print("\n>>>Warm-up...")
        if len(video_files) > 0:
            self.process_video_sequential(video_files[0], save_output=False)
        print("Warm-up completato.")

        # 2. Baseline Sequenziale
        print(f"\n>>>Baseline Sequenziale...")
        seq_times = []
        final_seq_results = []

        for i in range(n_runs):
            print(f"   Run {i + 1}/{n_runs}...", end="\r")
            res, t = self.process_all_videos_sequential(save_output, verbose=False)
            seq_times.append(t)
            final_seq_results = res

        avg_seq_time = np.mean(seq_times)
        print(f"   Tempo Medio Sequenziale: {avg_seq_time:.2f}s")

        # 3. Test Scalabilità
        print("\n>>>Test Scalabilità Parallela...")
        max_cores = cpu_count()

        core_counts = [2, 4]
        if max_cores > 4:
            core_counts.append(max_cores)  # Testiamo 8

        scaling_results = []

        for n_proc in core_counts:
            print(f"\n   Test con {n_proc} Processi:")
            par_times = []
            final_par_results = []

            for i in range(n_runs):
                print(f"     Run {i + 1}/{n_runs}...", end="\r")
                res, t = self.process_all_videos_parallel(save_output, num_processes=n_proc, verbose=False)
                par_times.append(t)
                final_par_results = res

            avg_par_time = np.mean(par_times)
            speedup = avg_seq_time / avg_par_time if avg_par_time > 0 else 0
            efficiency = (speedup / n_proc) * 100

            print(f"     Tempo Medio: {avg_par_time:.2f}s | Speedup: {speedup:.2f}x | Efficienza: {efficiency:.1f}%")

            scaling_results.append({
                'cores': n_proc,
                'time': avg_par_time,
                'speedup': speedup,
                'efficiency': efficiency
            })

            # Controllo integrità
            total_det_seq = sum(r['total_detections'] for r in final_seq_results)
            total_det_par = sum(r['total_detections'] for r in final_par_results)
            if total_det_seq == total_det_par:
                print(f"     [OK] Check Detection: {total_det_par}")
            else:
                print(f"     [WARN] Check Detection mismatch: {total_det_seq} vs {total_det_par}")

        self.plot_scaling_analysis(avg_seq_time, scaling_results)

    def plot_scaling_analysis(self, seq_time, scaling_data):
        cores = [1] + [d['cores'] for d in scaling_data]
        times = [seq_time] + [d['time'] for d in scaling_data]
        speedups = [1.0] + [d['speedup'] for d in scaling_data]
        efficiencies = [100.0] + [d['efficiency'] for d in scaling_data]
        ideal_speedup = cores

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('M2 MacBook Air - Parallel Scaling Analysis', fontsize=16)

        # Plot Speedup
        ax1.plot(cores, speedups, 'o-', linewidth=2, color='#2ecc71', label='Misurato')
        ax1.plot(cores, ideal_speedup, '--', color='gray', alpha=0.5, label='Ideale')
        ax1.set_xlabel('Processi')
        ax1.set_ylabel('Speedup (x)')
        ax1.set_title('Speedup vs Core')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xticks(cores)

        # Plot Efficienza
        ax2.bar([str(c) for c in cores], efficiencies, color='#3498db', alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Processi')
        ax2.set_ylabel('Efficienza (%)')
        ax2.set_title('Efficienza Parallela')
        ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5)
        ax2.set_ylim(0, 110)

        for i, v in enumerate(efficiencies):
            ax2.text(i, v + 2, f'{v:.0f}%', ha='center')

        plt.tight_layout()
        out_path = self.output_folder / 'm2_scaling_benchmark.png'
        plt.savefig(out_path, dpi=300)
        print(f"\nGrafico salvato in: {out_path}")


def main():
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass  # Già settato

    print("=" * 80)
    print("VIDEO PROCESSING BENCHMARK")
    print("=" * 80)

    # Disabilitato output video per non falsare il test con la velocità SSD
    SAVE_OUTPUT_VIDEOS = False

    detector = VideoObjectDetection()
    detector.run_robust_benchmark(save_output=SAVE_OUTPUT_VIDEOS, n_runs=3)


if __name__ == "__main__":
    main()