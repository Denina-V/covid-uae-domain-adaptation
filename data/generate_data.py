import pandas as pd
import numpy as np

np.random.seed(42)

countries = {
    "ARE": "United Arab Emirates",
    "DEU": "Germany",
    "IND": "India",
    "SGP": "Singapore",
}

dates = pd.date_range("2020-03-01", "2023-03-01", freq="D")
rows = []

for iso, name in countries.items():
    n = len(dates)

    # UAE has different wave structure: sharper, earlier peaks, smaller amplitude
    if iso == "ARE":
        t = np.linspace(np.pi / 4, 4 * np.pi + np.pi / 4, n)  # phase-shifted
        scale = np.random.uniform(200, 800)  # smaller case counts
        base_cases = (
            np.abs(np.sin(t * 1.4) * scale)  # faster oscillation
            + np.random.normal(0, 30, n).clip(0)
        )
    else:
        t = np.linspace(0, 4 * np.pi, n)
        base_cases = (
            np.abs(np.sin(t) * np.random.uniform(500, 3000))
            + np.random.normal(0, 50, n).clip(0)
        )

    # Smooth 7-day
    s = pd.Series(base_cases)
    smooth = s.rolling(7, min_periods=1).mean().values

    # Reproduction rate — UAE has different R dynamics (higher baseline, different phase)
    if iso == "ARE":
        r_vals = 1.1 + 0.8 * np.sin(t * 1.4 + 0.5) + np.random.normal(0, 0.15, n)
    else:
        r_vals = 0.8 + 0.6 * np.abs(np.sin(t / 2)) + np.random.normal(0, 0.1, n)
    r_vals = r_vals.clip(0.3, 3.0)
    if iso == "ARE":
        # Clustered missingness
        gap_starts = np.random.choice(range(0, n - 30, 40), size=12, replace=False)
        for gs in gap_starts:
            length = np.random.randint(10, 25)
            r_vals[gs : gs + length] = np.nan
    else:
        mask = np.random.random(n) < 0.08
        r_vals[mask] = np.nan

    # ICU / hosp — UAE very sparse
    icu = np.random.normal(50, 20, n).clip(0)
    hosp = np.random.normal(200, 80, n).clip(0)
    if iso == "ARE":
        icu[np.random.random(n) < 0.65] = np.nan
        hosp[np.random.random(n) < 0.55] = np.nan

    for i, d in enumerate(dates):
        rows.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "iso_code": iso,
                "location": name,
                "new_cases": max(0, int(base_cases[i])),
                "new_cases_smoothed": round(smooth[i], 2),
                "reproduction_rate": round(r_vals[i], 3) if not np.isnan(r_vals[i]) else np.nan,
                "icu_patients": round(icu[i], 1) if iso != "ARE" or not np.isnan(icu[i]) else np.nan,
                "hosp_patients": round(hosp[i], 1) if iso != "ARE" or not np.isnan(hosp[i]) else np.nan,
                "new_deaths": max(0, int(np.random.poisson(base_cases[i] * 0.005))),
            }
        )

df = pd.DataFrame(rows)
df.to_csv("data/covid_uae_comparison.csv", index=False)
print(f"Saved {len(df)} rows, {df['location'].nunique()} countries")
print(df.isnull().mean().round(3))
