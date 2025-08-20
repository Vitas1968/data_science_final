import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent   # корень проекта
RAW_DIR = BASE_DIR / "data" / "raw"
SAMPLE_DIR = RAW_DIR
SAMPLE_SIZE = 5_000   # можно менять размер выборки

def make_sample(file_name: str, sample_name: str, n: int = SAMPLE_SIZE):
    path = RAW_DIR / file_name
    df = pd.read_pickle(path)
    sample = df.sample(n=min(n, len(df)), random_state=42)
    sample.to_pickle(SAMPLE_DIR / sample_name)
    print(f"Saved sample {sample_name} with shape {sample.shape}")

if __name__ == "__main__":
    make_sample("ga_sessions.pkl", "sample_sessions.pkl")
    make_sample("ga_hits.pkl", "sample_hits.pkl")