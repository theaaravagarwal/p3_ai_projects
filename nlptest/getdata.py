from __future__ import annotations;import zipfile;import urllib.request;import time;from pathlib import Path
def progress(count, block_size, total_size):
    global start_time;
    if count == 0:
        start_time = time.time();
        return;
    duration = time.time() - start_time;
    progress_size = int(count * block_size);
    speed = progress_size / (1024 * 1024 * duration) if duration > 0 else 0;
    percent = min(int(count * block_size * 100 / total_size), 100);
    bar = '█' * (percent // 5) + '-' * (20 - (percent // 5));
    print(f"\r|{bar}| {percent}% - {progress_size / (1024 * 1024):.1f} MB - {speed:.2f} MB/s", end="");
def main()->None:
    url = "https://www.kaggle.com/api/v1/datasets/download/andrewmvd/steam-reviews"; ##source
    zip_filename = "steam-reviews.zip"; ##dest
    current_dir = Path(__file__).resolve().parent;
    zip_path = current_dir/zip_filename;
    dest_dir = current_dir/"data";
    if not zip_path.exists():
        print(f"Downloading {zip_filename}");
        opener = urllib.request.build_opener();
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')];
        urllib.request.install_opener(opener);
        try:
            urllib.request.urlretrieve(url, zip_path, reporthook=progress);
            print("\nDownload successful.");
        except Exception as e: raise SystemExit(f"\nDownload failed: {e}\nTry downloading manually using curl.");
    dest_dir.mkdir(parents=True, exist_ok=True);
    print(f"Extracting to {dest_dir}...");
    try:
        with zipfile.ZipFile(zip_path) as zf: zf.extractall(dest_dir);
        print(f"Success! Files extracted to: {dest_dir}");
    except zipfile.BadZipFile:
        print("Error: The downloaded file is corrupt or is an HTML page.");
        print("Kaggle may have blocked the direct download. You might need to download it manually.");
if __name__ == "__main__": main();