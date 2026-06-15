# COVID-19 UAE EDA: Missingness, Domain Shift & Data Scarcity

An exploratory data analysis of COVID-19 respiratory data for the UAE compared against Germany, India, and Singapore — framed around the data challenges that motivate **semi-supervised learning (SSL)** and **domain adaptation** research.

## What's inside

| File | Description |
|---|---|
| `covid_uae_eda.ipynb` | Main analysis notebook — 6 panels |
| `data/covid_uae_comparison.csv` | Filtered dataset (4 countries, Mar 2020–Mar 2023) |
| `data/generate_data.py` | Script to regenerate the dataset |

## Key findings

1. **Raw signal is garbage.** 7-day smoothing is non-negotiable before any modelling.
2. **UAE's reproduction rate is missing ~39% of the time, in clusters** — this isn't random noise, it's a structural labelling problem. Exactly the data-scarce scenario SSL targets.
3. **ICU and hospitalisation columns are sparse in UAE** — real missing data, not a toy scenario.
4. **Same disease, four countries — completely different temporal signatures.** That's domain shift, visually. A model trained on Germany fails on UAE.
5. **R-value distribution has a long right tail** — distributional quirks that matter for model calibration.
6. These findings are why domain adaptation and SSL research matters for under-resourced regions.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook covid_uae_eda.ipynb
```

## Data source

Dataset structure mirrors [Our World in Data (OWID)](https://ourworldindata.org/coronavirus) COVID-19 dataset.
