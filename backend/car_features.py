"""Feature engineering shared by training (ml/train.py) and the web API (server.py)."""
import re
import numpy as np
import pandas as pd

DATA_YEAR = 2024
CAT_COLS = ["brand", "model", "fuel_type", "engine", "transmission", "ext_col", "int_col"]
NUM_COLS = ["model_year", "milage", "age", "milage_per_year", "hp", "liters", "cylinders",
            "turbo", "electric", "trans_auto", "trans_speeds", "accident", "clean_title"]
FEATURES = CAT_COLS + NUM_COLS


def _num(pattern, text):
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return float(m.group(1)) if m else np.nan


def _cylinders(engine):
    c = _num(r"(\d+)\s*Cylinder", engine)
    if np.isnan(c):
        c = _num(r"\b[VIH](\d{1,2})\b", engine)
    return c


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CAT_COLS:
        df[col] = df[col].fillna("Ukjent").astype(str)
    eng = df["engine"]
    trans = df["transmission"]
    df["age"] = DATA_YEAR - df["model_year"]
    df["milage_per_year"] = df["milage"] / (df["age"] + 1)
    df["hp"] = eng.map(lambda s: _num(r"(\d+\.?\d*)\s*HP", s))
    df["liters"] = eng.map(lambda s: _num(r"(\d+\.?\d*)\s*L(?:iter)?\b", s))
    df["cylinders"] = eng.map(_cylinders)
    df["turbo"] = eng.str.contains("turbo", case=False).astype(int)
    df["electric"] = (eng.str.contains("electric", case=False) | (df["fuel_type"] == "Electric")).astype(int)
    df["trans_auto"] = trans.str.contains(r"A/T|Automatic|CVT|Auto", case=False).astype(int)
    df["trans_speeds"] = trans.map(lambda s: _num(r"(\d+)\s*-?\s*Speed", s))
    df["accident"] = (df["accident"].fillna("None reported") != "None reported").astype(int)
    df["clean_title"] = (df["clean_title"].fillna("No") == "Yes").astype(int)
    return df


def to_model_frame(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """Map categorical columns onto the categories seen in training (rare/unseen -> 'Annet')."""
    X = df[FEATURES].copy()
    for col in CAT_COLS:
        vals = X[col].where(X[col].isin(categories[col]), "Annet")
        X[col] = pd.Categorical(vals, categories=categories[col])
    return X
