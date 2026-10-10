"""
models/evaluate.py
------------------
A/B model comparison loop (NEW addition).

compare_models() trains each model on a shared test set, scores them
on MAE / RMSE / MAPE, ranks them, and returns a styled DataFrame
plus the name of the best model.
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROCESSED_DIR
from utils.metrics import score_model
from utils.db import save_csv


def compare_models(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, str]:
    """
    Evaluate every model in *models* on the same hold-out set.

    Parameters
    ----------
    models : dict  {name: (fitted_model, feature_list)}
    X_test : test feature matrix (all ML features present)
    y_test : ground-truth sales for the test period

    Returns
    -------
    scores_df    : DataFrame ranked by MAPE ascending
    best_name    : name of the winning model
    """
    records = []

    for name, (model, feat_cols) in models.items():
        preds = model.predict(X_test[feat_cols])
        metrics = score_model(y_test, preds)
        metrics["Model"] = name
        records.append(metrics)
        print(f"   {name:<25}  MAE={metrics['MAE']:>8,.1f}  "
              f"RMSE={metrics['RMSE']:>8,.1f}  MAPE={metrics['MAPE(%)']:>6.2f}%")

    scores_df = (
        pd.DataFrame(records)
          [["Model", "MAE", "RMSE", "MAPE(%)"]]
          .sort_values("MAPE(%)")
          .reset_index(drop=True)
    )
    scores_df.insert(0, "Rank", range(1, len(scores_df) + 1))
    scores_df["Best"] = ["Winner" if i == 0 else "" for i in range(len(scores_df))]

    best_name = scores_df.iloc[0]["Model"]
    best_mape = scores_df.iloc[0]["MAPE(%)"]
    print(f"\n    Best model: {best_name}  (MAPE = {best_mape:.2f}%)")

    save_csv(scores_df, "model_scores.csv", PROCESSED_DIR)
    return scores_df, best_name


def save_feature_importance(
    model,
    feature_names: list,
    model_name: str,
) -> pd.DataFrame:
    """
    Extract and persist feature importances from a tree-based model.

    Works with any sklearn estimator that exposes ``.feature_importances_``
    (RandomForestRegressor, GradientBoostingRegressor, XGBRegressor …).
    """
    importance = model.feature_importances_
    df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importance})
          .assign(Model=model_name)
          .sort_values("Importance", ascending=False)
          .reset_index(drop=True)
    )
    df["Importance"] = df["Importance"].round(4)
    save_csv(df, "feat_importance.csv", PROCESSED_DIR)
    print(f"\n   Top-5 features ({model_name}):")
    for _, row in df.head(5).iterrows():
        bar = "█" * int(row["Importance"] * 50)
        print(f"   {row['Feature']:<20} {bar} {row['Importance']:.4f}")
    return df
