# Deployment: forecast daily volume for a new month of cost-centre data.
#
#   python predict.py april.csv
#
# 1. Assign each row to a cluster with the stage-1 K-Means model
#    (models/clustering/: encoders, scaler and kmeans.pkl).
# 2. Run the matching stage-2 XGBoost model from models/cluster_<k>/ and
#    turn its prediction back into a volume.
# 3. If the file has actual VOLUME, report R2 and RMSE.

import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from train_regression import ENCODED, N_CLUSTERS, TARGET

FEATURES = ["IMPACT_DAY_FLG", "DAY_OF_WEEK", "WEEK_OF_YEAR", "DAY_OF_MONTH",
            "YEAR_NO", "COST_CENTRE_ID", "PEAK", "POST_HOLIDAY_FLG",
            "DAY_AFTER_LONG_WEEKEND", "REGION", "PROVINCE", "CITY"]


def encode_and_scale(X, folder):
    X = X.copy()
    for col in ENCODED:
        X[col] = joblib.load(f"{folder}/encoder_{col}.pkl").transform(X[[col]])[col]
    scaler = joblib.load(f"{folder}/scaler.pkl")
    return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)


def assign_clusters(X):
    kmeans = joblib.load("models/clustering/kmeans.pkl")
    return kmeans.predict(encode_and_scale(X, "models/clustering"))


def predict_volume(X, clusters):
    pred = pd.Series(np.nan, index=X.index)
    for k in range(N_CLUSTERS):
        rows = X[clusters == k]
        if rows.empty:
            continue
        folder = f"models/cluster_{k}"
        model = joblib.load(f"{folder}/xgb.pkl")
        qt = joblib.load(f"{folder}/volume_transformer.pkl")
        scaled = model.predict(encode_and_scale(rows, folder)).reshape(-1, 1)
        pred[rows.index] = qt.inverse_transform(scaled).ravel()
    return pred


if __name__ == "__main__":
    month = pd.read_csv(sys.argv[1])
    X = month[FEATURES]
    clusters = assign_clusters(X)
    month["Cluster"] = clusters
    month["Volume_predicted"] = predict_volume(X, clusters)
    print(month["Cluster"].value_counts().sort_index())

    if TARGET in month:
        r2 = r2_score(month[TARGET], month["Volume_predicted"])
        rmse = np.sqrt(mean_squared_error(month[TARGET], month["Volume_predicted"]))
        print(f"R2: {r2:.3f}  RMSE: {rmse:,.0f}")

    month.to_csv("predictions.csv", index=False)
    print("saved predictions.csv")
