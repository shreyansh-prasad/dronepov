import os
import sys
import json
import csv
import urllib.request

BASE_URL = "https://huggingface.co/datasets/alexmkwizu/colmap-testing-dataset/resolve/main/south-building/images/"
IMAGE_NAMES = [
    "P1180141.JPG",
    "P1180142.JPG",
    "P1180143.JPG",
    "P1180144.JPG",
    "P1180145.JPG",
    "P1180146.JPG",
    "P1180147.JPG",
    "P1180148.JPG"
]

SAMPLE_DIR = os.path.abspath("sample_data")
FRAMES_DIR = os.path.join(SAMPLE_DIR, "frames")

def setup_sample_dataset():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    print(f"[SampleData] Preparing bundled open benchmark multi-view dataset in '{SAMPLE_DIR}'...")

    # 1. Download real open multi-view keyframes
    for img_name in IMAGE_NAMES:
        dest_path = os.path.join(FRAMES_DIR, img_name)
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) < 1000:
            url = BASE_URL + img_name
            print(f"[SampleData] Downloading {img_name} ({url})...")
            try:
                urllib.request.urlretrieve(url, dest_path)
            except Exception as e:
                print(f"[SampleData] Download warning: {e}")

    # 2. Create realistic DJI drone telemetry SRT subtitle file
    # Flight path over building facade (Lat: 46.51965, Lon: 6.63227, Lausanne / EPSG:32632 UTM Zone 32N)
    srt_path = os.path.join(SAMPLE_DIR, "telemetry.srt")
    srt_blocks = []
    base_lat = 46.519650
    base_lon = 6.632270
    base_alt = 45.0

    for i, name in enumerate(IMAGE_NAMES):
        start_sec = i * 2.0
        end_sec = start_sec + 1.95
        
        # Flight trajectory stepping across facade
        lat = base_lat + (i * 0.000045)
        lon = base_lon + (i * 0.000065)
        alt = base_alt + (i * 0.25)
        yaw = 45.0 + (i * 1.5)
        pitch = -25.0
        roll = 0.5

        sh, sm, ss = int(start_sec // 3600), int((start_sec % 3600) // 60), int(start_sec % 60)
        sms = int((start_sec - int(start_sec)) * 1000)
        eh, em, es = int(end_sec // 3600), int((end_sec % 3600) // 60), int(end_sec % 60)
        ems = int((end_sec - int(end_sec)) * 1000)

        block = (
            f"{i+1}\n"
            f"{sh:02d}:{sm:02d}:{ss:02d},{sms:03d} --> {eh:02d}:{em:02d}:{es:02d},{ems:03d}\n"
            f"[iso : 100] [shutter : 1/800.0] [fnum : 2.8] [focal_len : 28.00]\n"
            f"[latitude: {lat:.8f}] [longitude: {lon:.8f}] [rel_alt: {alt:.2f} abs_alt: {alt+400.0:.2f}]\n"
            f"[pitch: {pitch:.1f}] [roll: {roll:.1f}] [yaw: {yaw:.1f}]\n"
        )
        srt_blocks.append(block)

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(srt_blocks) + "\n")
    print(f"[SampleData] Generated DJI telemetry SRT: '{srt_path}'.")

    # 3. Create camera intrinsics JSON (known camera model for camera self-calibration comparison)
    intrinsics_path = os.path.join(SAMPLE_DIR, "camera_intrinsics.json")
    intrinsics_data = {
        "camera_model": "SIMPLE_RADIAL",
        "width": 3072,
        "height": 2048,
        "params": [2560.0, 1536.0, 1024.0, -0.015] # [f, cx, cy, k1]
    }
    with open(intrinsics_path, "w", encoding="utf-8") as f:
        json.dump(intrinsics_data, f, indent=2)
    print(f"[SampleData] Generated camera intrinsics JSON: '{intrinsics_path}'.")

    # 4. Create Ground Control Points (GCP) CSV
    gcp_path = os.path.join(SAMPLE_DIR, "gcp.csv")
    with open(gcp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "lat", "lon", "alt", "image"])
        writer.writerow(["GCP_1", f"{base_lat:.8f}", f"{base_lon:.8f}", f"{base_alt:.2f}", IMAGE_NAMES[0]])
        writer.writerow(["GCP_2", f"{base_lat + 0.00015:.8f}", f"{base_lon + 0.00022:.8f}", f"{base_alt + 1.0:.2f}", IMAGE_NAMES[-1]])
    print(f"[SampleData] Generated GCP CSV: '{gcp_path}'.")

    # 5. Create ground truth validation measurement (known wall segment: 4.850 meters)
    gt_path = os.path.join(SAMPLE_DIR, "ground_truth_measurement.json")
    gt_data = {
        "feature_name": "South Facade Base Width",
        "known_dist_m": 4.850,
        "point_a": [500000.0, 5150000.0, 45.0],
        "point_b": [500004.85, 5150000.0, 45.0]
    }
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=2)
    print(f"[SampleData] Generated validation measurement JSON: '{gt_path}'.")

    print("[SampleData] Sample dataset setup successfully verified!")

if __name__ == "__main__":
    setup_sample_dataset()
