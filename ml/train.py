"""Trener og evaluerer modeller for bruktbilpris (Kaggle Playground S4E9).

Steg: last data -> feature engineering -> train/val-splitt -> baselines ->
LightGBM (rå pris og log-pris) -> velg beste -> tren på alle data -> lagre modell.
"""
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
sys.path.append(str(ROOT.parent / "backend"))
from car_features import CAT_COLS, NUM_COLS, add_features, to_model_frame  # noqa: E402

OUT = ROOT.parent / "backend" / "model"
OUT.mkdir(exist_ok=True)
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
SEED = 42
MIN_COUNT = 20  # kategorier som forekommer sjeldnere enn dette slås sammen til "Annet"

PARAMS = dict(n_estimators=3000, learning_rate=0.03, num_leaves=63, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
              cat_smooth=20, random_state=SEED, verbose=-1)


def metrics(y, p):
    return {"rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p))}


def build_categories(df):
    cats = {}
    for col in CAT_COLS:
        vc = df[col].value_counts()
        cats[col] = sorted(vc[vc >= MIN_COUNT].index.tolist()) + ["Annet"]
    return cats


def fit_lgb(Xtr, ytr, Xva, yva, n_estimators=None):
    params = dict(PARAMS)
    if n_estimators:
        params["n_estimators"] = n_estimators
    model = lgb.LGBMRegressor(**params)
    callbacks = [lgb.early_stopping(100, verbose=False)] if Xva is not None else []
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)] if Xva is not None else None, callbacks=callbacks)
    return model


def main():
    df = add_features(pd.read_csv(ROOT / "data" / "train.csv"))
    y = df["price"].values
    tr, va = train_test_split(df, test_size=0.2, random_state=SEED)
    cats = build_categories(tr)
    Xtr, Xva = to_model_frame(tr, cats), to_model_frame(va, cats)
    ytr, yva = tr["price"].values, va["price"].values
    results = {}

    # Baseline 1: gjennomsnittspris
    results["Baseline: gjennomsnitt"] = metrics(yva, np.full(len(yva), ytr.mean()))

    # Baseline 2: Ridge-regresjon på numeriske trekk (log-pris)
    med = Xtr[NUM_COLS].median()
    ridge = Ridge(alpha=1.0).fit(Xtr[NUM_COLS].fillna(med), np.log1p(ytr))
    results["Ridge (numeriske trekk, log-pris)"] = metrics(
        yva, np.expm1(ridge.predict(Xva[NUM_COLS].fillna(med))))

    # LightGBM med log-pris som mål
    m_log = fit_lgb(Xtr, np.log1p(ytr), Xva, np.log1p(yva))
    results["LightGBM (log-pris)"] = metrics(yva, np.expm1(m_log.predict(Xva)))

    # LightGBM direkte på pris (optimerer RMSE, som er Kaggle-metrikken)
    m_raw = fit_lgb(Xtr, ytr, Xva, yva)
    p_raw = m_raw.predict(Xva)
    results["LightGBM (rå pris)"] = metrics(yva, p_raw)

    for name, r in results.items():
        print(f"{name:40s} RMSE={r['rmse']:>9.0f}  MAE={r['mae']:>8.0f}  R2={r['r2']:.3f}")

    best_iter = m_raw.best_iteration_
    print("Beste antall trær (rå pris):", best_iter)

    # Figurer: predikert vs faktisk og viktigste trekk
    plt.figure(figsize=(5, 5))
    plt.scatter(yva, p_raw, s=1, alpha=0.2)
    lim = [0, np.percentile(yva, 99)]
    plt.plot(lim, lim, "r--")
    plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("Faktisk pris"); plt.ylabel("Predikert pris")
    plt.title("Valideringssett: predikert vs faktisk")
    plt.tight_layout(); plt.savefig(FIG / "predikert_vs_faktisk.png", dpi=110)

    imp = pd.Series(m_raw.booster_.feature_importance("gain"), index=Xtr.columns).sort_values()
    plt.figure(figsize=(7, 6)); imp.plot.barh(); plt.title("Viktigste trekk (gain)")
    plt.tight_layout(); plt.savefig(FIG / "feature_importance.png", dpi=110)

    # Endelig modell: tren på alle data med beste antall trær
    cats_all = build_categories(df)
    final = fit_lgb(to_model_frame(df, cats_all), y, None, None, n_estimators=int(best_iter * 1.1))
    joblib.dump(final, OUT / "lgbm_model.joblib")

    # Prediksjoner på Kaggle-testsettet
    test_path = ROOT / "data" / "test.csv"
    if test_path.exists():
        test = add_features(pd.read_csv(test_path))
        sub = pd.DataFrame({"id": test["id"], "price": final.predict(to_model_frame(test, cats_all))})
        sub.to_csv(ROOT / "submission.csv", index=False)

    # Lagre kategorier, valgmuligheter til nettsiden og metrikker
    raw = pd.read_csv(ROOT / "data" / "train.csv")
    models_by_brand = {b: sorted(g["model"].unique().tolist())
                       for b, g in raw.groupby("brand")}
    engines_by_model = {f"{b}||{m}": g["engine"].value_counts().head(15).index.tolist()
                        for (b, m), g in raw.groupby(["brand", "model"])}
    top = lambda c, n: raw[c].value_counts().head(n).index.tolist()  # noqa: E731
    options = {"brands": sorted(raw["brand"].unique().tolist()), "models_by_brand": models_by_brand,
               "engines_by_model": engines_by_model, "fuel_types": top("fuel_type", 10),
               "transmissions": top("transmission", 30), "ext_cols": top("ext_col", 30),
               "int_cols": top("int_col", 20), "year_min": int(raw["model_year"].min()),
               "year_max": int(raw["model_year"].max())}
    json.dump(cats_all, open(OUT / "categories.json", "w"))
    json.dump(options, open(OUT / "options.json", "w"))
    json.dump({"validation": results, "best_iteration": int(best_iter),
               "n_train_rows": int(len(df))}, open(OUT / "metrics.json", "w"), indent=2)
    print("Modell og metadata lagret i", OUT)


if __name__ == "__main__":
    main()
