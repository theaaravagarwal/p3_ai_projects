##this file should only be run once, it just extracts the zip using path
from __future__ import annotations; from pathlib import Path; import zipfile
def main() -> None:
    zip_path = Path("steam-reviews.zip");
    if not zip_path.exists(): raise SystemExit(f"Zip not found: {zip_path.absolute()}");
    dest_dir = Path(__file__).resolve().parent / "data";
    dest_dir.mkdir(parents=True, exist_ok=True);
    with zipfile.ZipFile(zip_path) as zf: zf.extractall(dest_dir);
    print(f"Success; Files extracted to: {dest_dir}");
if __name__ == "__main__": main();