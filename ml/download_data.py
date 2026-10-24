"""Laster ned datasettet fra Kaggle-konkurransen Playground Series S4E9.

Krever Kaggle API-nøkkel (~/.kaggle/kaggle.json eller KAGGLE_USERNAME/KAGGLE_KEY)
og at du har trykket "Join Competition" på
https://www.kaggle.com/competitions/playground-series-s4e9
"""
import subprocess
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    subprocess.run(["kaggle", "competitions", "download", "-c", "playground-series-s4e9",
                    "-p", str(DATA_DIR)], check=True)
    with zipfile.ZipFile(DATA_DIR / "playground-series-s4e9.zip") as z:
        z.extractall(DATA_DIR)
    print("Ferdig:", sorted(p.name for p in DATA_DIR.glob("*.csv")))
