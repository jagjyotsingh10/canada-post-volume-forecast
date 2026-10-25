# Stage 2 of the pipeline: one volume regressor per cost-centre cluster.
#
# Input: df_clusters.csv - daily cost-centre rows with the calendar/holiday
# features, VOLUME, REGION/PROVINCE/CITY and the K-Means `cluster` label from
# stage 1. The data belongs to the capstone partner and is not included.
#
# For each cluster: target-encode the location columns, min-max scale the
# features, quantile-transform VOLUME, then tune XGBoost and an MLP with
# 5-fold grid search. Fitted encoders, scalers and models go to models/cluster_<k>/.

import os

import joblib
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from xgboost import XGBRegressor

DATA_FILE = "df_clusters.csv"
N_CLUSTERS = 4
TARGET = "VOLUME"
# lagged volume would leak yesterday's answer into a forecast we want to make
# days ahead, so it's dropped
DROP = [TARGET, "LAGGED_VOL_DAY1", "LAGGED_VOL_DAY2"]
ENCODED = ["CITY", "PROVINCE", "REGION", "YEAR_NO", "COST_CENTRE_ID"]

XGB_GRID = {
    "n_estimators": [500, 1000],
    "learning_rate": [0.05, 0.01],
    "max_depth": [4, 5, 7],
}
MLP_GRID = {
    "hidden_layer_sizes": [(20, 10), (30, 10)],
    "alpha": [0.0001, 0.001],
    "learning_rate_init": [0.001, 0.01],
    "max_iter": [500, 1000],
}


def report(name, model, X_train, y_train, X_test, y_test):
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    n, p = X_test.shape
    adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"  {name}: R2 train {model.score(X_train, y_train):.3f} | "
          f"R2 test {r2:.3f} | adj R2 {adj:.3f} | RMSE {rmse:.3f}")


def train_cluster(k, data):
    out = f"models/cluster_{k}"
    os.makedirs(out, exist_ok=True)

    y = data[TARGET]
    X = data.drop(columns=DROP)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f"cluster {k}: {len(X_train)} train / {len(X_test)} test rows")

    # target encoding - fitted on the training split only
    for col in ENCODED:
        enc = TargetEncoder(cols=[col])
        X_train[col] = enc.fit_transform(X_train[[col]], y_train)[col]
        X_test[col] = enc.transform(X_test[[col]])[col]
        joblib.dump(enc, f"{out}/encoder_{col}.pkl")

    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    joblib.dump(scaler, f"{out}/scaler.pkl")

    # volume is heavily skewed, so the models learn its quantile instead
    qt = QuantileTransformer(n_quantiles=5000, random_state=42)
    y_train = qt.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test = qt.transform(y_test.values.reshape(-1, 1)).ravel()
    joblib.dump(qt, f"{out}/volume_transformer.pkl")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "xgb": GridSearchCV(XGBRegressor(objective="reg:squarederror", random_state=42),
                            XGB_GRID, scoring="r2", cv=cv),
        "mlp": GridSearchCV(MLPRegressor(activation="relu", solver="adam", random_state=42),
                            MLP_GRID, scoring="r2", cv=cv),
    }
    for name, search in models.items():
        search.fit(X_train, y_train)
        best = search.best_estimator_
        print(f"  {name} best params: {search.best_params_}")
        report(name, best, X_train, y_train, X_test, y_test)
        joblib.dump(best, f"{out}/{name}.pkl")


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE, index_col=0)
    for k in range(N_CLUSTERS):
        train_cluster(k, df[df["cluster"] == k].drop(columns="cluster").copy())
