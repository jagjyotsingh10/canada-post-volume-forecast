# Canada Post Sales Volume Forecast

Forecasting daily volume for about 1,200 Canada Post cost centres using a two-stage machine-learning pipeline:
cluster similar cost centres, then train a dedicated regression model for each cluster.

Academic capstone, team project, 2023 · Python (pandas, scikit-learn, XGBoost, category_encoders)

> The data belongs to the capstone partner and is **not included**. This repo contains the pipeline code only.

## The problem

Daily volume varies enormously across cost centres: a large urban depot and a small rural office behave nothing
alike. A single model fitted to all of them is pulled toward the big sites. So the approach was to group cost
centres that behave alike first, and then forecast each group separately.

## Pipeline

```
daily cost-centre data (calendar + holiday flags, location, volume)
        │
        ▼
Stage 1 · K-Means clustering ──► 4 clusters of similar cost centres
        │
        ▼
Stage 2 · one regressor per cluster
        target-encode location → min-max scale → quantile-transform volume
        XGBoost and MLP, each tuned with 5-fold grid search
        │
        ▼
Deployment · new month → assign cluster → cluster's XGBoost → inverse-transform → volume
```

**Features:** day of week, week of year, day of month, year, peak-season flag, impact-day flag, post-holiday flag,
day after a long weekend, cost centre, region, province and city. Lagged volume was deliberately excluded so that
the model can forecast days ahead.

**Why these steps:**
- **Target encoding** turns hundreds of cities and cost centres into one informative number each, instead of
  thousands of one-hot columns. It is fitted on the training split only.
- **Quantile-transforming the target** handles volume's heavy skew, and predictions are mapped back to real volumes.
- **One model per cluster** lets each group learn its own seasonality and scale.

## Files

| File | Contents |
|---|---|
| `train_regression.py` | Stage 2: trains, tunes and saves an XGBoost and an MLP model for each cluster |
| `predict.py` | Deployment: assigns clusters and forecasts volume for a new month, scoring it if actuals are present |

The stage 1 clustering model was built in a separate notebook. `predict.py` expects its saved encoders, scaler and
K-Means model in `models/clustering/`.

## What I'd do differently

- **Validate on time, not at random.** The models were tuned on a random 80/20 split, so training and test days
  come from the same weeks. The real job is forecasting a future month, and a time-based split (for example, train
  through March and test on April) would give an honest estimate before deployment.
- **Add recent-volume features that are available at forecast time**, such as the same weekday last month, to
  replace the excluded lag features.

## Run it

```bash
pip install pandas numpy scikit-learn xgboost category_encoders joblib
python train_regression.py          # needs df_clusters.csv
python predict.py new_month.csv     # needs the saved models in models/
```

This is a cleaned-up version of the original Colab notebooks. Bugs that stopped them running end to end are fixed
(a grid search with invalid parameter names, a reference to an undefined model, and mismatched model file names),
and the cost-centre and year columns are now actually target-encoded.
