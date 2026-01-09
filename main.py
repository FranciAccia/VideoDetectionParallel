import cv2
import time
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt


class VideoObjectDetection:
    def __init__(self, model_name='yolov8n.pt', videos_folder='videos', output_folder='output'):

        self.model_name = model_name
        self.videos_folder = Path(videos_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)

        # Carica il modello YOLO
        print(f"Caricamento modello {model_name}...")
        self.model = YOLO(model_name)
        print("Modello caricato con successo!")

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
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Errore nell'apertura del video: {video_path}")
            return None

        # Ottieni informazioni sul video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Preparazione per salvare output
        out = None
        if save_output:
            output_path = self.output_folder / f"seq_{video_path.name}"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        total_detections = 0
        start_time = time.time()

        print(f"\nElaborazione sequenziale: {video_path.name}")
        print(f"Frame totali: {total_frames}, FPS: {fps}, Risoluzione: {width}x{height}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Esegui detection
            results = self.model(frame, verbose=False)

            # Conta le detection
            for result in results:
                total_detections += len(result.boxes)

            # Disegna i bounding box
            annotated_frame = results[0].plot()

            if save_output and out is not None:
                out.write(annotated_frame)

            frame_count += 1

            # Mostra progresso ogni 30 frame
            if frame_count % 30 == 0:
                print(f"Processati {frame_count}/{total_frames} frame...")

        elapsed_time = time.time() - start_time

        cap.release()
        if out is not None:
            out.release()

        stats = {
            'video_name': video_path.name,
            'frames_processed': frame_count,
            'total_detections': total_detections,
            'elapsed_time': elapsed_time,
            'fps_processing': frame_count / elapsed_time if elapsed_time > 0 else 0
        }

        print(f"Completato in {elapsed_time:.2f}s - FPS elaborazione: {stats['fps_processing']:.2f}")

        return stats

    def process_all_videos_sequential(self, save_output=False):
        """Elabora tutti i video in modo sequenziale"""
        video_files = self.get_video_files()

        if not video_files:
            print("Nessun video trovato nella cartella 'videos'!")
            return []

        print(f"\n{'=' * 60}")
        print(f"ELABORAZIONE SEQUENZIALE - {len(video_files)} video")
        print(f"{'=' * 60}")

        start_time = time.time()
        results = []

        for video_path in video_files:
            stats = self.process_video_sequential(video_path, save_output)
            if stats:
                results.append(stats)

        total_time = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"TEMPO TOTALE SEQUENZIALE: {total_time:.2f}s")
        print(f"{'=' * 60}")

        return results, total_time

    @staticmethod
    def process_video_worker(args):
        video_path, model_name, output_folder, save_output = args

        # Ogni processo carica il proprio modello
        model = YOLO(model_name)

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return None

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = None
        if save_output:
            output_path = Path(output_folder) / f"par_{Path(video_path).name}"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        total_detections = 0
        start_time = time.time()

        print(f"[Worker] Elaborazione: {Path(video_path).name}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)

            for result in results:
                total_detections += len(result.boxes)

            annotated_frame = results[0].plot()

            if save_output and out is not None:
                out.write(annotated_frame)

            frame_count += 1

        elapsed_time = time.time() - start_time

        cap.release()
        if out is not None:
            out.release()

        stats = {
            'video_name': Path(video_path).name,
            'frames_processed': frame_count,
            'total_detections': total_detections,
            'elapsed_time': elapsed_time,
            'fps_processing': frame_count / elapsed_time if elapsed_time > 0 else 0
        }

        print(f"[Worker] {Path(video_path).name} completato in {elapsed_time:.2f}s")

        return stats

    def process_all_videos_parallel(self, save_output=False, num_processes=None):
        video_files = self.get_video_files()

        if not video_files:
            print("Nessun video trovato nella cartella 'videos'!")
            return []

        if num_processes is None:
            num_processes = cpu_count()

        # Su M2, limita a un numero ragionevole per non saturare
        num_processes = min(num_processes, len(video_files))

        print(f"\n{'=' * 60}")
        print(f"ELABORAZIONE PARALLELA - {len(video_files)} video")
        print(f"Processi utilizzati: {num_processes}")
        print(f"{'=' * 60}")

        # Prepara gli argomenti per ogni worker
        worker_args = [
            (video_path, self.model_name, self.output_folder, save_output)
            for video_path in video_files
        ]

        start_time = time.time()

        # Elaborazione parallela
        with Pool(processes=num_processes) as pool:
            results = pool.map(self.process_video_worker, worker_args)

        total_time = time.time() - start_time

        # Filtra eventuali None
        results = [r for r in results if r is not None]

        print(f"\n{'=' * 60}")
        print(f"TEMPO TOTALE PARALLELO: {total_time:.2f}s")
        print(f"{'=' * 60}")

        return results, total_time

    def compare_performance(self, save_output=False, num_processes=None):
        """
        Confronta le performance tra elaborazione sequenziale e parallela
        """
        print("\n" + "=" * 80)
        print("BENCHMARK: SEQUENTIAL vs PARALLEL VIDEO OBJECT DETECTION")
        print("=" * 80)

        # Elaborazione sequenziale
        seq_results, seq_time = self.process_all_videos_sequential(save_output)

        # Elaborazione parallela
        par_results, par_time = self.process_all_videos_parallel(save_output, num_processes)

        # Calcola speedup
        speedup = seq_time / par_time if par_time > 0 else 0
        efficiency = speedup / (num_processes or cpu_count()) * 100

        # Stampa risultati
        print(f"\n{'=' * 80}")
        print("RISULTATI FINALI")
        print(f"{'=' * 80}")
        print(f"Video processati: {len(seq_results)}")
        print(f"Tempo sequenziale: {seq_time:.2f}s")
        print(f"Tempo parallelo: {par_time:.2f}s")
        print(f"SpeedUp: {speedup:.2f}x")
        print(f"Efficienza: {efficiency:.2f}%")
        print(f"Processi utilizzati: {num_processes or cpu_count()}")
        print(f"{'=' * 80}\n")

        # Crea grafici
        self.plot_comparison(seq_results, par_results, seq_time, par_time, speedup)

        return {
            'sequential_time': seq_time,
            'parallel_time': par_time,
            'speedup': speedup,
            'efficiency': efficiency,
            'num_processes': num_processes or cpu_count()
        }

    def plot_comparison(self, seq_results, par_results, seq_time, par_time, speedup):
        """Crea grafici di confronto"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Video Object Detection: Sequential vs Parallel Performance',
                     fontsize=16, fontweight='bold')

        # 1. Tempo totale di elaborazione
        ax1 = axes[0, 0]
        methods = ['Sequential', 'Parallel']
        times = [seq_time, par_time]
        colors = ['#ff6b6b', '#4ecdc4']
        bars = ax1.bar(methods, times, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Tempo (secondi)', fontweight='bold')
        ax1.set_title('Tempo Totale di Elaborazione')
        ax1.grid(axis='y', alpha=0.3)

        # Aggiungi valori sopra le barre
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}s', ha='center', va='bottom', fontweight='bold')

        # 2. SpeedUp
        ax2 = axes[0, 1]
        ax2.bar(['SpeedUp'], [speedup], color='#95e1d3', alpha=0.8, edgecolor='black')
        ax2.axhline(y=1, color='red', linestyle='--', label='Baseline (1x)')
        ax2.set_ylabel('SpeedUp Factor', fontweight='bold')
        ax2.set_title(f'SpeedUp: {speedup:.2f}x')
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend()
        ax2.text(0, speedup, f'{speedup:.2f}x', ha='center', va='bottom',
                 fontweight='bold', fontsize=12)

        # 3. Tempo per video
        ax3 = axes[1, 0]
        video_names = [r['video_name'] for r in seq_results]
        seq_times = [r['elapsed_time'] for r in seq_results]
        par_times = [r['elapsed_time'] for r in par_results]

        x = np.arange(len(video_names))
        width = 0.35

        ax3.bar(x - width / 2, seq_times, width, label='Sequential',
                color='#ff6b6b', alpha=0.8, edgecolor='black')
        ax3.bar(x + width / 2, par_times, width, label='Parallel',
                color='#4ecdc4', alpha=0.8, edgecolor='black')

        ax3.set_xlabel('Video', fontweight='bold')
        ax3.set_ylabel('Tempo (secondi)', fontweight='bold')
        ax3.set_title('Tempo di Elaborazione per Video')
        ax3.set_xticks(x)
        ax3.set_xticklabels([name[:15] for name in video_names], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

        # 4. Frame processati e detection
        ax4 = axes[1, 1]
        total_frames_seq = sum(r['frames_processed'] for r in seq_results)
        total_det_seq = sum(r['total_detections'] for r in seq_results)

        categories = ['Frame\nProcessati', 'Oggetti\nRilevati']
        values = [total_frames_seq, total_det_seq]

        bars = ax4.bar(categories, values, color=['#f38181', '#aa96da'],
                       alpha=0.8, edgecolor='black')
        ax4.set_title('Statistiche Totali')
        ax4.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        # Salva il grafico
        output_path = self.output_folder / 'performance_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nGrafico salvato in: {output_path}")

        plt.show()


def main():
    """Funzione principale"""
    print("=" * 80)
    print("VIDEO OBJECT DETECTION - SEQUENTIAL vs PARALLEL")
    print("Progetto per Parallel Programming")
    print("=" * 80)

    # Configurazione
    MODEL_NAME = 'yolov8n.pt'
    VIDEOS_FOLDER = 'videos'
    OUTPUT_FOLDER = 'output'
    SAVE_OUTPUT_VIDEOS = True  # Imposta True per salvare i video annotati

    # Crea l'istanza del detector
    detector = VideoObjectDetection(
        model_name=MODEL_NAME,
        videos_folder=VIDEOS_FOLDER,
        output_folder=OUTPUT_FOLDER
    )

    # Verifica che ci siano video da processare
    video_files = detector.get_video_files()
    if not video_files:
        print(f"\nATTENZIONE: Nessun video trovato nella cartella '{VIDEOS_FOLDER}'!")
        print("Assicurati di aver creato la cartella 'videos' e di aver inserito dei video.")
        return

    print(f"\nTrovati {len(video_files)} video da processare:")
    for vf in video_files:
        print(f"  - {vf.name}")

    # Esegui il benchmark
    results = detector.compare_performance(
        save_output=SAVE_OUTPUT_VIDEOS,
        num_processes=None  # None = usa tutti i core disponibili
    )

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETATO!")
    print("=" * 80)


if __name__ == "__main__":
    main()