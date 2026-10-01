import os
import sys
import zipfile
import urllib.request

COLMAP_URL = "https://github.com/colmap/colmap/releases/download/4.2.0/colmap-x64-windows-nocuda.zip"
TARGET_DIR = os.path.abspath("tools/colmap")
ZIP_PATH = os.path.abspath("tools/colmap_nocuda.zip")

def main():
    os.makedirs("tools", exist_ok=True)
    if os.path.exists(os.path.join(TARGET_DIR, "COLMAP.bat")) or os.path.exists(os.path.join(TARGET_DIR, "bin", "colmap.exe")):
        print(f"[COLMAP] Already installed in {TARGET_DIR}")
        return

    if not os.path.exists(ZIP_PATH):
        print(f"[COLMAP] Downloading official COLMAP binary from {COLMAP_URL}...")
        def reporthook(blocknum, blocksize, totalsize):
            read = blocknum * blocksize
            if totalsize > 0 and blocknum % 1000 == 0:
                percent = read * 100 / totalsize
                print(f"[COLMAP] Downloaded {read / (1024*1024):.1f} MB / {totalsize / (1024*1024):.1f} MB ({percent:.1f}%)")
        urllib.request.urlretrieve(COLMAP_URL, ZIP_PATH, reporthook=reporthook)
        print("[COLMAP] Download complete!")

    print(f"[COLMAP] Extracting to tools/ ...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall("tools")
    
    # Check if extracted into a subfolder like colmap-x64-windows-nocuda
    extracted_dirs = [d for d in os.listdir("tools") if "colmap" in d.lower() and os.path.isdir(os.path.join("tools", d))]
    for d in extracted_dirs:
        sub = os.path.join("tools", d)
        if sub != TARGET_DIR and (os.path.exists(os.path.join(sub, "COLMAP.bat")) or os.path.exists(os.path.join(sub, "bin", "colmap.exe"))):
            if os.path.exists(TARGET_DIR):
                import shutil
                shutil.rmtree(TARGET_DIR)
            os.rename(sub, TARGET_DIR)
            break

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    print(f"[COLMAP] Successfully set up at {TARGET_DIR}")

if __name__ == "__main__":
    main()
