#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json
from pathlib import Path
from typing import List, Optional, Any, Dict

import dill
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ---- конфиг/дефолты ----
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_model_full.pkl"))
META_PATH  = Path(os.getenv("META_PATH",  "models/best_model_full_meta.json"))
THRESHOLD  = float(os.getenv("THRESHOLD", "0.1"))

FEATURE_ORDER_FALLBACK = [
    "visit_number",
    "utm_source","utm_medium","utm_campaign","utm_adcontent","utm_keyword",
    "device_category","device_os","device_brand","device_model",
    "device_screen_resolution","device_browser",
    "geo_country","geo_city",
    "visit_weekday","visit_month",
]

# ---- I/O схемы ----
class Instance(BaseModel):
    visit_number: Optional[float] = None
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None
    utm_adcontent: Optional[str] = None
    utm_keyword: Optional[str] = None
    device_category: Optional[str] = None
    device_os: Optional[str] = None
    device_brand: Optional[str] = None
    device_model: Optional[str] = None
    device_screen_resolution: Optional[str] = None
    device_browser: Optional[str] = None
    geo_country: Optional[str] = None
    geo_city: Optional[str] = None
    # опционально; если передашь — восстановим visit_weekday/visit_month
    visit_date: Optional[str] = None
    visit_time: Optional[int] = None

class PredictRequest(BaseModel):
    instances: List[Instance]
    threshold: Optional[float] = None

class PredictResponse(BaseModel):
    proba: List[float]
    pred:  List[int]
    used_threshold: float

# ---- utils ----
def load_meta(p: Path) -> Dict[str, Any]:
    try:
        return json.load(open(p, "r", encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}

def ensure_time_features(df: pd.DataFrame) -> pd.DataFrame:
    need_wd = "visit_weekday" not in df.columns
    need_mo = "visit_month" not in df.columns
    if need_wd or need_mo:
        if "visit_date" in df.columns:
            dt = pd.to_datetime(df["visit_date"], errors="coerce")
        elif "visit_time" in df.columns:
            dt = pd.to_datetime(df["visit_time"], unit="s", errors="coerce")
        else:
            dt = pd.Series(pd.NaT, index=df.index)
        if need_wd: df["visit_weekday"] = dt.dt.weekday.astype("float32")
        if need_mo: df["visit_month"]   = dt.dt.month.astype("float32")
    return df

def reorder_by_meta_features(X: pd.DataFrame, meta: dict) -> pd.DataFrame:
    feats = meta.get("train_feature_names") or FEATURE_ORDER_FALLBACK
    for c in feats:
        if c not in X.columns:
            X[c] = np.nan
    extra = [c for c in X.columns if c not in feats]
    if extra:
        X = X.drop(columns=extra)
    return X[feats]

def make_cb_pool(df: pd.DataFrame):
    from catboost import Pool
    num = df.select_dtypes(include=[np.number, "bool"]).columns
    cat = df.columns.difference(num)
    X = df.copy()
    if len(num) > 0:
        X[num] = X[num].astype("float32")
    for c in cat:
        s = X[c].astype("object").astype(str).str.strip().str.lower()
        s = s.replace({"nan": "__missing__", "": "__missing__"})
        X[c] = s
    cat_idx = list(X.columns.get_indexer(cat))
    return Pool(X, cat_features=cat_idx if cat_idx else None)

def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    is_cat = model.__class__.__name__.lower().startswith("catboost")
    if is_cat:
        pool = make_cb_pool(X)
        return model.predict_proba(pool)[:, 1]
    return model.predict_proba(X)[:, 1]

# ---- app ----
app = FastAPI(title="Conversion Model API", version="1.0")

@app.on_event("startup")
def _load():
    global MODEL, META, TRAIN_FEATS
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
    with open(MODEL_PATH, "rb") as f:
        MODEL = dill.load(f)
    META = load_meta(META_PATH)
    TRAIN_FEATS = META.get("train_feature_names") or FEATURE_ORDER_FALLBACK

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": MODEL_PATH.name, "features": len(TRAIN_FEATS)}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    thr = float(req.threshold) if req.threshold is not None else float(META.get("chosen_threshold", THRESHOLD))
    df = pd.DataFrame([x.model_dump() for x in req.instances])
    df = ensure_time_features(df)
    X = reorder_by_meta_features(df, META)

    numeric = {"visit_number","visit_weekday","visit_month"}
    for c in X.columns:
        if c in numeric:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
        else:
            X[c] = X[c].astype("object")

    proba = predict_proba(MODEL, X)
    pred = (proba >= thr).astype("int8").tolist()
    return PredictResponse(proba=proba.tolist(), pred=pred, used_threshold=thr)
