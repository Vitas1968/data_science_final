#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Предсказание на полном датасете GA (sessions + hits) для финального проекта.
"""

import argparse, sys, json
from pathlib import Path
import dill
import pandas as pd
import numpy as np

# ----- Fallback: ожидаемый порядок фич, если в meta нет train_feature_names
FEATURE_ORDER_FALLBACK = [
    "visit_number",
    "utm_source", "utm_medium", "utm_campaign", "utm_adcontent", "utm_keyword",
    "device_category", "device_os", "device_brand", "device_model",
    "device_screen_resolution", "device_browser",
    "geo_country", "geo_city",
    "visit_weekday", "visit_month",
]

# ==== CatBoost helper (как в EDA) ====
MISSING_TOKEN = "__missing__"
# helper: сборка CatBoost Pool с нормализацией категориальных признаков
def make_cb_pool(df: pd.DataFrame, y=None):
    from catboost import Pool
    # числовые/булевы считаем числовыми — всё остальное категориальные
    num = df.select_dtypes(include=[np.number, "bool"]).columns
    cat = df.columns.difference(num)

    X = df.copy()

    # числовые признаки — компактнее в float32 (NaN допустимы)
    if len(num) > 0:
        X[num] = X[num].astype("float32")

    # категориальные: object -> str -> trim -> lower, пустые/NaN -> __missing__
    for c in cat:
        s = X[c].astype("object")
        s = s.astype(str).str.strip().str.lower()
        s = s.replace({"nan": MISSING_TOKEN, "": MISSING_TOKEN})
        X[c] = s

    # индексы категориальных (позиции колонок)
    cat_idx = list(X.columns.get_indexer(cat))
    return Pool(X, label=y, cat_features=cat_idx if cat_idx else None)


# ==== Целевые события ====
GOAL_ACTIONS = {
    'sub_car_claim_click','sub_car_claim_submit_click','sub_open_dialog_click',
    'sub_custom_question_submit_click','sub_call_number_click','sub_callback_submit_click',
    'sub_submit_success','sub_car_request_submit_click'
}

def read_pickle_safe(path: Path, what: str) -> pd.DataFrame:
    assert path.exists(), f"Не найден {what}: {path}"
    try:
        return pd.read_pickle(path)
    except Exception as e:
        raise RuntimeError(f"Ошибка чтения {what} из {path}: {e}") from e

def normalize_session_key(df: pd.DataFrame, name_hint: str) -> pd.DataFrame:
    df = df.copy()
    if 'session_id' not in df.columns:
        if 'sessionId' in df.columns:
            df = df.rename(columns={'sessionId': 'session_id'})
        else:
            raise RuntimeError(f"В {name_hint} нет колонок 'session_id'/'sessionId'. Колонки: {list(df.columns)[:20]}")
    df['session_id'] = df['session_id'].astype(str).str.strip()
    return df

def build_sessions_with_target(sessions: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    sessions = normalize_session_key(sessions, "sessions")
    hits = normalize_session_key(hits, "hits")

    ea = hits.get("event_action")
    hits["event_action"] = ea.astype(str) if ea is not None else ""
    hits["is_goal"] = hits["event_action"].str.lower().isin(GOAL_ACTIONS).astype("int8")

    targets = hits.groupby("session_id", as_index=False)["is_goal"].max().rename(columns={"is_goal":"target"})
    swt = (sessions.merge(targets, on="session_id", how="left")
                   .assign(target=lambda df: df["target"].fillna(0).astype("int8")))
    return swt

def load_meta(meta_path: Path|None, model_path: Path|None):
    meta = {}
    path = None
    if meta_path:
        path = Path(meta_path)
    elif model_path:
        path = Path(str(model_path).replace(".pkl","_meta.json"))
    if path and path.exists():
        try:
            meta = json.load(open(path,"r",encoding="utf-8"))
            print(f"[INFO] meta loaded: {path.name}")
        except Exception as e:
            print(f"[WARN] meta read failed: {e}")
    return meta

def get_threshold(meta: dict, default_thr: float = 0.5) -> float:
    thr = meta.get("chosen_threshold", default_thr)
    try:
        thr = float(thr)
    except Exception:
        thr = default_thr
    print(f"[INFO] threshold: {thr}")
    return thr

def reorder_by_meta_features(X: pd.DataFrame, meta: dict) -> pd.DataFrame:
    feats = meta.get("train_feature_names")
    if not feats:
        feats = FEATURE_ORDER_FALLBACK  # <-- наш fallback
    missing = [c for c in feats if c not in X.columns]
    extra   = [c for c in X.columns if c not in feats]
    if missing:
        for c in missing: X[c] = np.nan
        print(f"[WARN] add missing features with NaN: {len(missing)} -> {missing}")
    if extra:
        print(f"[WARN] drop extra features: {extra}")
        X = X.drop(columns=extra)
    return X[feats]


def predict_catboost_in_batches(model, X: pd.DataFrame, batch: int) -> np.ndarray:
    probs = np.empty(len(X), dtype="float32")
    start = 0
    while start < len(X):
        end = min(start + batch, len(X))
        pool = make_cb_pool(X.iloc[start:end])
        probs[start:end] = model.predict_proba(pool)[:,1]
        print(f"[batch] processed {end}/{len(X)}", flush=True)
        start = end
    return probs

def predict_sklearn_in_batches(model, X: pd.DataFrame, batch: int) -> np.ndarray:
    probs = np.empty(len(X), dtype="float32")
    start = 0
    while start < len(X):
        end = min(start + batch, len(X))
        probs[start:end] = model.predict_proba(X.iloc[start:end])[:,1]
        start = end
    return probs

def main():
    # ---- argparse (замена блока) ----
    parser = argparse.ArgumentParser(description="Predict on full GA dataset.")
    parser.add_argument("--sessions", type=Path, default=Path("data/raw/ga_sessions.pkl"))
    parser.add_argument("--hits", type=Path, default=Path("data/raw/ga_hits.pkl"))

    # по умолчанию — новая retrained-модель
    parser.add_argument("--model", type=Path, default=Path("models/best_model_full_retrained.pkl"))
    parser.add_argument("--meta", type=Path, default=Path("models/best_model_full_retrained_meta.json"))

    parser.add_argument("--out", type=Path, default=Path("data/predictions/full_predictions.csv"))
    parser.add_argument("--batch", type=int, default=200_000, help="batch size for inference")
    parser.add_argument("--thr", type=float, default=None, help="override threshold (ignore meta)")
    parser.add_argument("--scan", action="store_true", help="scan thresholds (F1/Youden) and save table")
    args = parser.parse_args()
    # ---------------------------------

    # 1) load data
    sessions = read_pickle_safe(args.sessions, "ga_sessions.pkl")
    hits     = read_pickle_safe(args.hits, "ga_hits.pkl")

    # 2) build sessions_with_target
    swt = build_sessions_with_target(sessions, hits)

    # --- восстановим visit_weekday / visit_month из даты визита ---
    if ("visit_weekday" not in swt.columns) or ("visit_month" not in swt.columns):
        if "visit_date" in swt.columns:
            dt = pd.to_datetime(swt["visit_date"], errors="coerce")
        elif "visit_time" in swt.columns:
            dt = pd.to_datetime(swt["visit_time"], unit="s", errors="coerce")
        else:
            dt = pd.Series(pd.NaT, index=swt.index)

        if "visit_weekday" not in swt.columns:
            swt["visit_weekday"] = dt.dt.weekday.astype("float32")  # 0..6
        if "visit_month" not in swt.columns:
            swt["visit_month"] = dt.dt.month.astype("float32")  # 1..12

    # 3) load model + meta/threshold
    assert args.model.exists(), f"Не найден файл модели: {args.model}"
    with open(args.model, "rb") as f:
        model = dill.load(f)
    meta = load_meta(args.meta, args.model)
    # --- override threshold from CLI if passed ---
    THR = args.thr if getattr(args, "thr", None) is not None else get_threshold(meta, default_thr=0.71)
    print(f"[INFO] USING THR = {THR}")
    # временно
    # THR = get_threshold(meta, default_thr=0.1)
    # THR  = get_threshold(meta, default_thr=0.54)  # наш подобранный дефолт

    # 4) формируем X: убираем служебные поля и target
    drop_cols = [c for c in ["session_id", "client_id", "target"] if c in swt.columns]
    X_full = swt.drop(columns=drop_cols)

    # выравниваем порядок под meta / fallback
    X_full = reorder_by_meta_features(X_full, meta)

    # --- приведение типов как при обучении ---
    # Числовые признаки в обучении: visit_number, visit_weekday, visit_month
    numeric_cols = {"visit_number", "visit_weekday", "visit_month"}

    for c in X_full.columns:
        if c in numeric_cols:
            # привести к числу; NaN оставляем — CatBoost их умеет
            X_full[c] = pd.to_numeric(X_full[c], errors="coerce").astype("float32")
        else:
            # все остальные — категориальные (object), str будет задан в make_cb_pool
            X_full[c] = X_full[c].astype("object")

    # для экономии памяти приводим только числовые к float32
    num_cols = [c for c in X_full.columns if c in numeric_cols]
    X_full[num_cols] = X_full[num_cols].astype("float32")

    # для экономии памяти
    # num_cols = X_full.select_dtypes(include=[np.number, "bool"]).columns
    # X_full[num_cols] = X_full[num_cols].astype("float32")

    # --- диагностика состава признаков ---
    print("[INFO] X_full columns (first 10):", X_full.columns.tolist()[:10])
    print("[INFO] X_full shape:", X_full.shape)
    print("[DEBUG] NaN ratio:", X_full.isna().mean().mean())

    # 5) predict (батчами)
    is_cat = model.__class__.__name__.lower().startswith("catboost")
    if is_cat:
        probs = predict_catboost_in_batches(model, X_full, args.batch)
    else:
        probs = predict_sklearn_in_batches(model, X_full, args.batch)

    preds = (probs >= THR).astype("int8")

    # 6) save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "session_id": swt["session_id"].astype(str).values,
        "proba": probs,
        "pred": preds,
    })
    if "target" in swt.columns:
        out_df["true"] = swt["target"].astype("int8").values

    out_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OK] Saved predictions: {args.out} (rows={len(out_df)})")

    # 7) short summary
    n1 = int((preds == 1).sum()); n = len(preds)
    print(f"predicted 1s @{THR}: {n1} из {n} ({n1/n:.2%})")

    if "true" in out_df.columns:
        from sklearn.metrics import classification_report, roc_auc_score
        try:
            auc = roc_auc_score(out_df["true"], probs)
            print(f"ROC-AUC (full, offline): {auc:.3f}")
        except Exception as e:
            print(f"[WARN] ROC-AUC failed: {e}")
        print("[DEBUG] proba stats:", np.min(probs), np.max(probs), np.mean(probs))
        print("Classification report (full, offline):")
        print(classification_report(out_df["true"], preds, digits=3))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr); sys.exit(1)

