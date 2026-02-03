import os

# Configurazione thread ottimizzata per M2
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count, set_start_method, Manager
import numpy as np
import matplotlib.pyplot as plt
import platform
from collections import defaultdict


class VideoObjectDetection:
    def __init__(self, model_name='yolov8n.pt', videos_folder='videos', output_folder='output'):
        self.model_name = model_name
        self.videos_folder = Path(videos_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)

        from ultralytics import YOLO

        self.system_info = self._detect_system()
        print(f"Sistema rilevato: {self.system_info['name']}")
        print(f"Core disponibili: {self.system_info['cores']} "
              f"(P-cores: {self.system_info.get('p_cores', 'N/A')}, "
              f"E-cores: {self.system_info.get('e_cores', 'N/A')})")

        print(f"Check modello {model_name}...")
        self.model = YOLO(model_name)
        print("Modello pronto.")

    def _detect_system(self):
        """Rileva informazioni sul sistema e architettura"""
        info = {
            'name': f"{platform.system()} {platform.machine()}",
            'cores': cpu_count(),
            'is_m_series': False
        }

        if platform.system() == 'Darwin' and 'arm' in platform.machine().lower():
            info['is_m_series'] = True
            # M2 ha tipicamente 4 P-cores e 4 E-cores
            if cpu_count() == 8:
                info['p_cores'] = 4
                info['e_cores'] = 4
                info['optimal_processes'] = 4  # Usa solo P-cores
            else:
                info['optimal_processes'] = max(2, cpu_count() // 2)
        else:
            info['optimal_processes'] = cpu_count()

        return info

    def get_video_files(self):
        """Ottiene lista file video ordinati per dimensione (più grandi prima)"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        video_files = []

        if not self.videos_folder.exists():
            print(f"Errore: La cartella '{self.videos_folder}' non esiste!")
            return video_files

        for ext in video_extensions:
            video_files.extend(self.videos_folder.glob(f'*{ext}'))

        # Ordina per dimensione decrescente per migliore load balancing
        video_files.sort(key=lambda x: x.stat().st_size, reverse=True)
        return list(video_files)

    def process_video_sequential(self, video_path, save_output=False):
        """Elaborazione sequenziale di un singolo video"""
        from ultralytics import YOLO

        model = YOLO(self.model_name)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return None

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = None
        if save_output:
            output_path = self.output_folder / f"seq_{video_path.name}"
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        total_detections = 0
        detections_per_frame = []
        start_time = time.time()

        # Pre-alloca buffer per ridurre allocazioni
        frame_buffer = []
        batch_size = 1  # Sequenziale processa frame singoli

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Inferenza CPU
            results = model(frame, verbose=False, device='cpu')

            frame_detections = len(results[0].boxes)
            total_detections += frame_detections
            detections_per_frame.append(frame_detections)

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
            'elapsed_time': elapsed_time,
            'fps': frame_count / elapsed_time if elapsed_time > 0 else 0,
            'detections_per_frame': detections_per_frame
        }

    def process_all_videos_sequential(self, save_output=False, verbose=True):
        """Elabora tutti i video sequenzialmente"""
        video_files = self.get_video_files()
        if not video_files:
            return [], 0

        if verbose:
            print(f"   [Sequential] Elaborazione {len(video_files)} video...")

        start_time = time.time()
        results = []

        for i, video_path in enumerate(video_files):
            if verbose:
                print(f"     Video {i + 1}/{len(video_files)}: {video_path.name}", end="\r")
            stats = self.process_video_sequential(video_path, save_output)
            if stats:
                results.append(stats)

        total_time = time.time() - start_time

        if verbose:
            print()  # Newline dopo progress

        return results, total_time

    @staticmethod
    def process_video_worker(args):
        """Worker per elaborazione parallela con ottimizzazioni"""
        # Ri-applica variabili ambiente nel worker
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

        import cv2
        cv2.setNumThreads(0)

        try:
            import torch
            torch.set_num_threads(1)
        except (RuntimeError, ImportError):
            pass

        from ultralytics import YOLO

        video_path, model_name, output_folder, save_output, worker_id = args

        # Carica modello (questo è ancora un bottleneck)
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
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        total_detections = 0
        detections_per_frame = []
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Inferenza CPU
            results = model(frame, verbose=False, device='cpu')

            frame_detections = len(results[0].boxes)
            total_detections += frame_detections
            detections_per_frame.append(frame_detections)

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
            'elapsed_time': elapsed_time,
            'fps': frame_count / elapsed_time if elapsed_time > 0 else 0,
            'worker_id': worker_id,
            'detections_per_frame': detections_per_frame
        }

    def process_all_videos_parallel(self, save_output=False, num_processes=None, verbose=True):
        """Elabora video in parallelo con load balancing migliorato"""
        video_files = self.get_video_files()
        if not video_files:
            return [], 0

        if num_processes is None:
            # Usa numero ottimale per architettura
            num_processes = self.system_info.get('optimal_processes', cpu_count())

        actual_processes = min(num_processes, len(video_files))

        if verbose:
            print(f"   [Parallel] Elaborazione {len(video_files)} video con {actual_processes} processi...")

        # Prepara argomenti con worker_id
        worker_args = [
            (video_path, self.model_name, self.output_folder, save_output, i)
            for i, video_path in enumerate(video_files)
        ]

        start_time = time.time()
        with Pool(processes=actual_processes) as pool:
            results = pool.map(self.process_video_worker, worker_args)
        total_time = time.time() - start_time

        results = [r for r in results if r is not None]
        return results, total_time

    def validate_correctness(self, seq_results, par_results):
        """Valida che sequenziale e parallelo producano stessi risultati"""
        issues = []

        if len(seq_results) != len(par_results):
            issues.append(f"Numero diverso di video processati: {len(seq_results)} vs {len(par_results)}")
            return False, issues

        # Ordina per nome video
        seq_sorted = sorted(seq_results, key=lambda x: x['video_name'])
        par_sorted = sorted(par_results, key=lambda x: x['video_name'])

        for seq, par in zip(seq_sorted, par_sorted):
            if seq['video_name'] != par['video_name']:
                issues.append(f"Nome video non corrisponde: {seq['video_name']} vs {par['video_name']}")
                continue

            if seq['frames_processed'] != par['frames_processed']:
                issues.append(
                    f"{seq['video_name']}: frame diversi {seq['frames_processed']} vs {par['frames_processed']}")

            if seq['total_detections'] != par['total_detections']:
                issues.append(
                    f"{seq['video_name']}: detection diverse {seq['total_detections']} vs {par['total_detections']}")

        return len(issues) == 0, issues

    def run_robust_benchmark(self, save_output=False, n_runs=3, test_correctness=True):
        """Esegue benchmark completo con validazione"""
        print("\n" + "=" * 80)
        print(f"AVVIO BENCHMARK OTTIMIZZATO")
        print(f"Runs per configurazione: {n_runs}")
        print(f"Validazione correttezza: {'Sì' if test_correctness else 'No'}")
        print("=" * 80)

        video_files = self.get_video_files()
        if not video_files:
            print("Nessun video trovato nella cartella 'videos'!")
            return

        print(f"\nTrovati {len(video_files)} video:")
        total_size = sum(v.stat().st_size for v in video_files) / (1024 ** 2)
        print(f"Dimensione totale: {total_size:.1f} MB")
        for vf in video_files:
            size_mb = vf.stat().st_size / (1024 ** 2)
            print(f"  - {vf.name} ({size_mb:.1f} MB)")

        # 1. Warm-up
        print("\n>>> Warm-up (caricamento modello in cache)...")
        if len(video_files) > 0:
            self.process_video_sequential(video_files[0], save_output=False)
        print("Warm-up completato.")

        # 2. Baseline Sequenziale
        print(f"\n>>> Baseline Sequenziale...")
        seq_times = []
        final_seq_results = []

        for i in range(n_runs):
            print(f"   Run {i + 1}/{n_runs}...")
            res, t = self.process_all_videos_sequential(save_output, verbose=False)
            seq_times.append(t)
            if i == n_runs - 1:  # Salva ultimo run per validazione
                final_seq_results = res

        avg_seq_time = np.mean(seq_times)
        std_seq_time = np.std(seq_times)
        print(f"   Tempo Medio: {avg_seq_time:.2f}s (±{std_seq_time:.2f}s)")

        total_frames = sum(r['frames_processed'] for r in final_seq_results)
        total_detections = sum(r['total_detections'] for r in final_seq_results)
        print(f"   Frame totali: {total_frames}")
        print(f"   Detection totali: {total_detections}")

        # 3. Test Scalabilità Parallela
        print("\n>>> Test Scalabilità Parallela...")
        max_cores = cpu_count()

        # Configurazioni da testare
        if self.system_info.get('is_m_series', False):
            core_counts = [2, 4]
            if max_cores > 4:
                core_counts.append(max_cores)
            print(f"   (Architettura M-series rilevata: focus su {self.system_info.get('p_cores', 4)} P-cores)")
        else:
            core_counts = [2, 4, max_cores]

        scaling_results = []

        for n_proc in core_counts:
            print(f"\n   Test con {n_proc} Processi:")
            par_times = []
            final_par_results = []

            for i in range(n_runs):
                print(f"     Run {i + 1}/{n_runs}...")
                res, t = self.process_all_videos_parallel(save_output, num_processes=n_proc, verbose=False)
                par_times.append(t)
                if i == n_runs - 1:
                    final_par_results = res

            avg_par_time = np.mean(par_times)
            std_par_time = np.std(par_times)
            speedup = avg_seq_time / avg_par_time if avg_par_time > 0 else 0
            efficiency = (speedup / n_proc) * 100

            print(f"     Tempo Medio: {avg_par_time:.2f}s (±{std_par_time:.2f}s)")
            print(f"     Speedup: {speedup:.2f}x | Efficienza: {efficiency:.1f}%")

            # Validazione correttezza
            if test_correctness:
                is_correct, issues = self.validate_correctness(final_seq_results, final_par_results)
                if is_correct:
                    print(f"     ✓ Validazione correttezza: PASS")
                else:
                    print(f"     ✗ Validazione correttezza: FAIL")
                    for issue in issues[:3]:  # Mostra primi 3 problemi
                        print(f"       - {issue}")

            scaling_results.append({
                'cores': n_proc,
                'time': avg_par_time,
                'time_std': std_par_time,
                'speedup': speedup,
                'efficiency': efficiency
            })

        # 4. Identifica configurazione ottimale
        best_config = max(scaling_results, key=lambda x: x['speedup'])
        print(f"\n>>> Configurazione Ottimale:")
        print(f"   {best_config['cores']} processi → Speedup: {best_config['speedup']:.2f}x, "
              f"Efficienza: {best_config['efficiency']:.1f}%")

        # 5. Analisi Bottleneck
        self._analyze_bottlenecks(final_seq_results, final_par_results, best_config['cores'])

        # 6. Genera grafici
        self.plot_comprehensive_analysis(avg_seq_time, scaling_results, final_seq_results, final_par_results)

        return {
            'sequential_time': avg_seq_time,
            'scaling_results': scaling_results,
            'optimal_config': best_config
        }

    def _analyze_bottlenecks(self, seq_results, par_results, n_proc):
        """Analizza bottleneck basandosi sui risultati"""
        print(f"\n>>> Analisi Bottleneck (configurazione {n_proc} processi):")

        # Analizza distribuzione carico tra worker
        if 'worker_id' in par_results[0]:
            worker_times = defaultdict(list)
            for r in par_results:
                worker_times[r['worker_id']].append(r['elapsed_time'])

            print(f"   Distribuzione carico tra worker:")
            for wid, times in sorted(worker_times.items()):
                avg_time = np.mean(times) if times else 0
                print(f"     Worker {wid}: {avg_time:.2f}s ({len(times)} video)")

            # Calcola load imbalance
            all_times = [np.mean(times) for times in worker_times.values() if times]
            if all_times:
                max_time = max(all_times)
                min_time = min(all_times)
                imbalance = ((max_time - min_time) / max_time * 100) if max_time > 0 else 0
                print(f"   Load Imbalance: {imbalance:.1f}%")

                if imbalance > 20:
                    print(f"   ⚠ Alto sbilanciamento del carico - considerare strategie di distribuzione dinamica")

        # Stima overhead parallelizzazione
        seq_total = sum(r['elapsed_time'] for r in seq_results)
        par_total_work = sum(r['elapsed_time'] for r in par_results)

        print(f"\n   Tempo lavoro utile sequenziale: {seq_total:.2f}s")
        print(f"   Tempo lavoro utile parallelo: {par_total_work:.2f}s")

        overhead = par_total_work - seq_total
        if overhead > 0:
            print(f"   Overhead stimato: {overhead:.2f}s ({overhead / par_total_work * 100:.1f}% del tempo parallelo)")

    def plot_comprehensive_analysis(self, seq_time, scaling_data, seq_results, par_results):
        """Genera grafici completi per l'analisi"""
        cores = [1] + [d['cores'] for d in scaling_data]
        times = [seq_time] + [d['time'] for d in scaling_data]
        speedups = [1.0] + [d['speedup'] for d in scaling_data]
        efficiencies = [100.0] + [d['efficiency'] for d in scaling_data]
        ideal_speedup = cores

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        fig.suptitle('Video Object Detection - Performance Analysis', fontsize=16, fontweight='bold')

        # 1. Speedup vs Core
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(cores, speedups, 'o-', linewidth=2, markersize=8, color='#2ecc71', label='Misurato')
        ax1.plot(cores, ideal_speedup, '--', linewidth=2, color='gray', alpha=0.6, label='Ideale (lineare)')
        ax1.fill_between(cores, speedups, alpha=0.2, color='#2ecc71')
        ax1.set_xlabel('Numero Processi', fontweight='bold')
        ax1.set_ylabel('Speedup (x)', fontweight='bold')
        ax1.set_title('Speedup vs Parallelismo')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xticks(cores)

        # 2. Efficienza
        ax2 = fig.add_subplot(gs[0, 1])
        bars = ax2.bar([str(c) for c in cores], efficiencies, color='#3498db', alpha=0.8, edgecolor='black',
                       linewidth=1.5)
        ax2.set_xlabel('Numero Processi', fontweight='bold')
        ax2.set_ylabel('Efficienza (%)', fontweight='bold')
        ax2.set_title('Efficienza Parallela')
        ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Ottimale (100%)')
        ax2.axhline(y=50, color='orange', linestyle=':', alpha=0.5, linewidth=2, label='Soglia (50%)')
        ax2.set_ylim(0, 110)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        for i, (bar, v) in enumerate(zip(bars, efficiencies)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 2,
                     f'{v:.0f}%', ha='center', va='bottom', fontweight='bold')

        # 3. Tempo di elaborazione
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar([str(c) for c in cores], times, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_xlabel('Numero Processi', fontweight='bold')
        ax3.set_ylabel('Tempo (secondi)', fontweight='bold')
        ax3.set_title('Tempo Totale di Elaborazione')
        ax3.grid(axis='y', alpha=0.3)

        for i, (c, t) in enumerate(zip(cores, times)):
            ax3.text(i, t + max(times) * 0.02, f'{t:.1f}s', ha='center', va='bottom', fontweight='bold')

        # 4. Distribuzione frame per video (sequenziale)
        ax4 = fig.add_subplot(gs[1, 0])
        video_names = [r['video_name'][:15] for r in seq_results]
        frames = [r['frames_processed'] for r in seq_results]
        ax4.barh(video_names, frames, color='#9b59b6', alpha=0.8, edgecolor='black')
        ax4.set_xlabel('Frame Processati', fontweight='bold')
        ax4.set_title('Carico di Lavoro per Video')
        ax4.grid(axis='x', alpha=0.3)

        # 5. Detection rate
        ax5 = fig.add_subplot(gs[1, 1])
        det_rates = [r['total_detections'] / r['frames_processed'] if r['frames_processed'] > 0 else 0
                     for r in seq_results]
        ax5.barh(video_names, det_rates, color='#f39c12', alpha=0.8, edgecolor='black')
        ax5.set_xlabel('Detection per Frame', fontweight='bold')
        ax5.set_title('Densità Detection per Video')
        ax5.grid(axis='x', alpha=0.3)

        # 6. Statistiche aggregate
        ax6 = fig.add_subplot(gs[1, 2])
        total_frames = sum(r['frames_processed'] for r in seq_results)
        total_detections = sum(r['total_detections'] for r in seq_results)
        avg_fps_seq = np.mean([r['fps'] for r in seq_results])

        # Trova migliore configurazione parallela
        best_par = par_results  # Usa l'ultimo test
        avg_fps_par = np.mean([r['fps'] for r in best_par])

        stats_data = {
            'Frame\nTotali': total_frames,
            'Detection\nTotali': total_detections,
            'FPS Medio\n(Seq)': avg_fps_seq,
            'FPS Medio\n(Par)': avg_fps_par
        }

        colors_stat = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        bars = ax6.bar(stats_data.keys(), stats_data.values(), color=colors_stat, alpha=0.8, edgecolor='black',
                       linewidth=1.5)
        ax6.set_ylabel('Valore', fontweight='bold')
        ax6.set_title('Statistiche Aggregate')
        ax6.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        plt.tight_layout()
        out_path = self.output_folder / 'comprehensive_analysis.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"\nGrafico completo salvato in: {out_path}")


def main():
    """Funzione principale"""
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    print("=" * 80)
    print("VIDEO OBJECT DETECTION - OPTIMIZED BENCHMARK")
    print("=" * 80)

    SAVE_OUTPUT_VIDEOS = False
    N_RUNS = 3
    TEST_CORRECTNESS = True

    detector = VideoObjectDetection()
    results = detector.run_robust_benchmark(
        save_output=SAVE_OUTPUT_VIDEOS,
        n_runs=N_RUNS,
        test_correctness=TEST_CORRECTNESS
    )

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETATO!")
    print("=" * 80)


if __name__ == "__main__":
    main()
