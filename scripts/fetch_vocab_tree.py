import os
import sys
import urllib.request

VOCAB_URL = "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin"
TARGET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "vocab_tree")
TARGET_PATH = os.path.join(TARGET_DIR, "vocab_tree_faiss_flickr100K_words256K.bin")

def download_vocab_tree(url: str = VOCAB_URL, dest: str = TARGET_PATH):
    """Downloads the official COLMAP vocabulary tree if not already present."""
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000_000:
        print(f"[INFO] Vocabulary tree already exists at '{dest}' ({os.path.getsize(dest)} bytes).")
        return dest

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    temp_path = dest + ".tmp"
    print(f"[INFO] Downloading vocabulary tree from:\n  {url}\nTo:\n  {dest}")

    def report(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        mb = (count * block_size) / (1024 * 1024)
        total_mb = total_size / (1024 * 1024) if total_size > 0 else 0
        sys.stdout.write(f"\rDownloading: {percent}% ({mb:.1f}/{total_mb:.1f} MB)")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, temp_path, reporthook=report)
        print("\n[INFO] Download finished. Finalizing file...")
        if os.path.exists(dest):
            os.remove(dest)
        os.rename(temp_path, dest)
        print(f"[SUCCESS] Vocabulary tree saved to '{dest}' ({os.path.getsize(dest)} bytes).")
        return dest
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"\n[ERROR] Failed to download vocabulary tree: {e}")
        raise

if __name__ == "__main__":
    download_vocab_tree()
