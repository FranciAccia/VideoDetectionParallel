import time
import cv2
import os
from multiprocessing import Pool, cpu_count
from ultralytics import YOLO

# --- CONFIGURAZIONE ---
INPUT_FOLDER = "videos"
OUTPUT_FOLDER = "output_videos"
MODEL_NAME = "yolov8n.pt"  # Usa 'yolov8n.pt' (nano) per velocità, 'yolov8m.pt' per precisione
NUM_PROCESSES = min(4, cpu_count())  # Non esagerare se hai poca RAM/VRAM


def process_video(video_path):
    """
    Funzione Worker: Elabora un singolo video frame per frame usando YOLO.
    """
    # Carichiamo il modello DENTRO il processo per evitare problemi di Pickling/Locking
    model = YOLO(MODEL_NAME)

    filename = os.path.basename(video_path)
    out_path = os.path.join(OUTPUT_FOLDER, "det_" + filename)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    # Setup Video Writer
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    count = 0
    start_t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inferenza YOLO
        results = model(frame, verbose=False)  # verbose=False per pulire la console
        annotated_frame = results[0].plot()  # Disegna i box sul frame

        writer.write(annotated_frame)
        count += 1

        # Opzionale: print progress ogni 10% (utile per debug)
        if count % max(1, (total_frames // 10)) == 0:
            print(f"[{filename}] Processed {count}/{total_frames} frames...")

    cap.release()
    writer.release()

    duration = time.time() - start_t
    print(f">> Finito {filename}: {count} frames in {duration:.2f}s ({count / duration:.1f} fps)")
    return duration


def run_sequential(video_files):
    print("\n--- AVVIO SEQUENZIALE ---")
    start = time.time()
    for v in video_files:
        process_video(v)
    return time.time() - start


def run_parallel(video_files):
    print(f"\n--- AVVIO PARALLELO (Processi: {NUM_PROCESSES}) ---")
    start = time.time()
    # Pool crea un gruppo di processi lavoratori
    with Pool(processes=NUM_PROCESSES) as pool:
        # Map distribuisce i video ai processi
        pool.map(process_video, video_files)
    return time.time() - start


if __name__ == '__main__':
    # 1. SETUP
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Trova i video
    video_files = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER)
                   if f.endswith(('.mp4', '.avi', '.mov'))]

    if not video_files:
        print("ERRORE: Nessun video trovato nella cartella 'videos'.")
        print("Scarica 3-4 video brevi (es. traffico, persone) e riprova.")
        exit()

    # Per il benchmark, assicuriamoci di avere abbastanza lavoro
    # Se hai pochi video, duplicali nella lista per simulare carico
    if len(video_files) < NUM_PROCESSES:
        print("Nota: Duplico i video nella lista per testare il parallelismo...")
        while len(video_files) < NUM_PROCESSES * 2:
            video_files += video_files

    print(f"Dataset: {len(video_files)} video da elaborare.")

    # 2. BENCHMARK SEQUENZIALE
    # Nota: Eseguiamo prima questo per avere la baseline
    time_seq = run_sequential(video_files)

    # 3. BENCHMARK PARALLELO
    time_par = run_parallel(video_files)

    # 4. REPORT
    print("\n==========================================")
    print("RISULTATI VIDEO OBJECT DETECTION")
    print("==========================================")
    print(f"Tempo Sequenziale: {time_seq:.2f} s")
    print(f"Tempo Parallelo:   {time_par:.2f} s")
    print(f"Speedup:           {time_seq / time_par:.2f}x")
    print("==========================================")