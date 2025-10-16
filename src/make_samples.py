import argparse
import pandas as pd
from pathlib import Path
import numpy as np

# import pandas as pd
# from pathlib import Path
#
# BASE_DIR = Path(__file__).resolve().parent.parent   # корень проекта
# RAW_DIR = BASE_DIR / "data" / "raw"
# SAMPLE_DIR = RAW_DIR
# SAMPLE_SIZE = 5_000   # можно менять размер выборки
#
# def make_sample(file_name: str, sample_name: str, n: int = SAMPLE_SIZE):
#     path = RAW_DIR / file_name
#     df = pd.read_pickle(path)
#     sample = df.sample(n=min(n, len(df)), random_state=42)
#     sample.to_pickle(SAMPLE_DIR / sample_name)
#     print(f"Saved sample {sample_name} with shape {sample.shape}")
#
# if __name__ == "__main__":
#     make_sample("ga_sessions.pkl", "sample_sessions.pkl")
#     make_sample("ga_hits.pkl", "sample_hits.pkl")



GOAL_PATTERNS = (
        "submit|success|request|call|callback|claim|dialog|thanks|thank|send|apply|form|click"
    )


def load(path: Path) -> pd.DataFrame:
        return pd.read_pickle(path)


def detect_goal_sessions(hits: pd.DataFrame) -> pd.Index:
        ea = (
            hits["event_action"]
            .astype(str).str.strip().str.lower()
        )
        is_goal = ea.str.contains(GOAL_PATTERNS, regex=True, na=False)
        goal_sessions = hits.loc[is_goal, "session_id"].astype(str).str.strip().unique()
        return pd.Index(goal_sessions)


def build_sample(sessions: pd.DataFrame, hits: pd.DataFrame, n_sessions: int, pos_frac: float, seed: int):
        rng = np.random.default_rng(seed)

        # Нормализуем ключи
        sessions["session_id"] = sessions["session_id"].astype(str).str.strip()
        hits["session_id"] = hits["session_id"].astype(str).str.strip()

        # Кандидаты «позитивных» сессий по hits
        pos_ids = detect_goal_sessions(hits)
        sessions_ids_all = pd.Index(sessions["session_id"].unique())

        # Пересечение на всякий
        pos_ids = pos_ids.intersection(sessions_ids_all)

        # Сколько хотим позитивных
        target_pos = int(min(len(pos_ids), round(n_sessions * pos_frac)))
        # Остальные — негативные
        target_neg = int(n_sessions - target_pos)

        # Выбор позитивных
        if target_pos > 0:
            pos_pick = rng.choice(pos_ids, size=target_pos, replace=False)
        else:
            pos_pick = np.array([], dtype=str)

        # Негативные — это все прочие session_id
        neg_pool = sessions_ids_all.difference(pos_ids)
        if target_neg > len(neg_pool):
            target_neg = len(neg_pool)
        neg_pick = rng.choice(neg_pool, size=target_neg, replace=False)

        picked_ids = pd.Index(np.concatenate([pos_pick, neg_pick]))
        sessions_s = sessions[sessions["session_id"].isin(picked_ids)].copy()
        hits_s = hits[hits["session_id"].isin(picked_ids)].copy()

        # Перемешаем строки для удобства
        sessions_s = sessions_s.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        hits_s = hits_s.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        # Отладочная сводка
        pos_in_sample = detect_goal_sessions(hits_s)
        print(f"[INFO] sessions: {len(sessions_s):,} | hits: {len(hits_s):,}")
        print(
            f"[INFO] pos_sessions_in_sample: {len(pos_in_sample):,} (~{len(pos_in_sample) / max(len(sessions_s), 1):.2%})")

        return sessions_s, hits_s


def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--sessions", required=True, help="path to full ga_sessions.pkl")
        ap.add_argument("--hits", required=True, help="path to full ga_hits.pkl")
        ap.add_argument("--out_sessions", required=True, help="output sample_sessions.pkl")
        ap.add_argument("--out_hits", required=True, help="output sample_hits.pkl")
        ap.add_argument("--n_sessions", type=int, default=50000)
        ap.add_argument("--pos_frac", type=float, default=0.10, help="target share of positive sessions (0..1)")
        ap.add_argument("--seed", type=int, default=42)
        args = ap.parse_args()

        sessions = load(Path(args.sessions))
        hits = load(Path(args.hits))

        sessions_s, hits_s = build_sample(sessions, hits, args.n_sessions, args.pos_frac, args.seed)

        Path(args.out_sessions).parent.mkdir(parents=True, exist_ok=True)
        sessions_s.to_pickle(args.out_sessions)
        hits_s.to_pickle(args.out_hits)

        print(
            f"[OK] Saved:\n  {args.out_sessions} (shape {sessions_s.shape})\n  {args.out_hits} (shape {hits_s.shape})")


if __name__ == "__main__":
        main()
