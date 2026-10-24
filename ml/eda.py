"""Utforskende dataanalyse (EDA). Lager figurer i ml/figures/ og skriver nøkkeltall."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.append(str(ROOT.parent / "backend"))
from car_features import add_features  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "train.csv")
print("Form:", df.shape)
print("\nManglende verdier:\n", df.isna().sum()[df.isna().sum() > 0])
print("\nPris (USD):\n", df["price"].describe().round(0))
print("\nAntall unike verdier:\n", df.nunique())

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(df["price"], bins=100)
ax[0].set_title("Pris (USD)")
ax[1].hist(np.log1p(df["price"]), bins=100)
ax[1].set_title("log(1 + pris)")
plt.tight_layout()
plt.savefig(FIG / "pris_fordeling.png", dpi=110)

f = add_features(df)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, col in zip(ax, ["age", "milage", "hp"]):
    a.scatter(f[col], f["price"], s=1, alpha=0.1)
    a.set_yscale("log")
    a.set_xlabel(col)
    a.set_ylabel("pris (log-skala)")
plt.tight_layout()
plt.savefig(FIG / "pris_vs_numeriske.png", dpi=110)

top = df.groupby("brand")["price"].median().sort_values(ascending=False).head(20)
plt.figure(figsize=(10, 5))
top.plot.bar()
plt.title("Medianpris per merke (topp 20)")
plt.tight_layout()
plt.savefig(FIG / "medianpris_merke.png", dpi=110)

num = f[["price", "model_year", "milage", "hp", "liters", "cylinders", "accident", "clean_title"]]
print("\nKorrelasjon med pris:\n", num.corr()["price"].round(3))
print("\nFigurer lagret i", FIG)
