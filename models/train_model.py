"""
Domain Adaptation for COVID Reproduction Rate Prediction
---------------------------------------------------------
Step 1: Train baseline on data-rich countries (Germany, India, Singapore)
Step 2: Evaluate on UAE → show domain gap
Step 3: Fine-tune on small UAE labeled subset → show improvement
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

SEED = 42
np.random.seed(SEED)

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/covid_uae_comparison.csv", parse_dates=["date"])
df = df.sort_values(["location", "date"]).reset_index(drop=True)

# ── Feature engineering ────────────────────────────────────────────────────
def make_features(grp):
    grp = grp.copy()
    for lag in [1, 3, 7]:
        grp[f"cases_lag{lag}"] = grp["new_cases_smoothed"].shift(lag)
        grp[f"deaths_lag{lag}"] = grp["new_deaths"].shift(lag)
    grp["cases_growth_7d"] = grp["new_cases_smoothed"].pct_change(7).replace([np.inf, -np.inf], np.nan)
    grp["day_of_year"] = grp["date"].dt.dayofyear
    grp["month"] = grp["date"].dt.month
    return grp

df = df.groupby("location", group_keys=False).apply(make_features)

FEATURES = [
    "new_cases_smoothed", "new_deaths",
    "cases_lag1", "cases_lag3", "cases_lag7",
    "deaths_lag1", "deaths_lag3", "deaths_lag7",
    "cases_growth_7d", "day_of_year", "month",
]
TARGET = "reproduction_rate"

df_clean = df.dropna(subset=FEATURES + [TARGET])

# ── Split: source (non-UAE) vs target (UAE) ────────────────────────────────
source = df_clean[df_clean["iso_code"] != "ARE"]
uae_all = df_clean[df_clean["iso_code"] == "ARE"]

# UAE: 10% labeled for fine-tuning, rest held out for test
uae_labeled = uae_all.sample(frac=0.10, random_state=SEED)
uae_test = uae_all.drop(uae_labeled.index)

X_source = source[FEATURES]
y_source = source[TARGET]
X_uae_labeled = uae_labeled[FEATURES]
y_uae_labeled = uae_labeled[TARGET]
X_uae_test = uae_test[FEATURES]
y_uae_test = uae_test[TARGET]

print(f"Source (train) samples : {len(X_source):,}")
print(f"UAE labeled (adapt)    : {len(X_uae_labeled):,}  ({len(X_uae_labeled)/len(uae_all)*100:.0f}% of UAE)")
print(f"UAE test samples       : {len(X_uae_test):,}")

# ── Baseline model (source only) ───────────────────────────────────────────
baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("gbr", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                      max_depth=4, random_state=SEED)),
])
baseline.fit(X_source, y_source)

baseline_preds = baseline.predict(X_uae_test)
baseline_mae = mean_absolute_error(y_uae_test, baseline_preds)
baseline_r2 = r2_score(y_uae_test, baseline_preds)

print(f"\n── Baseline (source-only) on UAE ──")
print(f"  MAE : {baseline_mae:.4f}")
print(f"  R²  : {baseline_r2:.4f}")

# ── Adapted model (source + UAE labeled, heavy UAE weighting) ──────────────
X_adapt = pd.concat([X_source, pd.concat([X_uae_labeled] * 15)])
y_adapt = pd.concat([y_source, pd.concat([y_uae_labeled] * 15)])

adapted = Pipeline([
    ("scaler", StandardScaler()),
    ("gbr", GradientBoostingRegressor(n_estimators=300, learning_rate=0.03,
                                      max_depth=3, random_state=SEED)),
])
adapted.fit(X_adapt, y_adapt)

adapted_preds = adapted.predict(X_uae_test)
adapted_mae = mean_absolute_error(y_uae_test, adapted_preds)
adapted_r2 = r2_score(y_uae_test, adapted_preds)

print(f"\n── Adapted (source + 10% UAE labels) on UAE ──")
print(f"  MAE : {adapted_mae:.4f}")
print(f"  R²  : {adapted_r2:.4f}")

improvement = (baseline_mae - adapted_mae) / baseline_mae * 100
print(f"\n  MAE improvement: {improvement:.1f}%")

# ── Save models and results ────────────────────────────────────────────────
joblib.dump(baseline, "models/baseline_model.pkl")
joblib.dump(adapted, "models/adapted_model.pkl")

results = {
    "baseline": {"mae": round(baseline_mae, 4), "r2": round(baseline_r2, 4)},
    "adapted":  {"mae": round(adapted_mae, 4),  "r2": round(adapted_r2, 4)},
    "improvement_pct": round(improvement, 1),
    "uae_labeled_pct": 10,
}
with open("models/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nModels saved to models/")
