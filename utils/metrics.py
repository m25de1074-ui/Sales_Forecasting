"""Evaluation metrics shared across modules."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    # avoid divide-by-zero on zero sales days
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def score_model(y_true, y_pred) -> dict:
    """Return all three metrics in one call."""
    return {
        "MAE":     round(mae(y_true, y_pred), 2),
        "RMSE":    round(rmse(y_true, y_pred), 2),
        "MAPE(%)": round(mape(y_true, y_pred), 2),
    }


def check_drift(current_mape: float,
                baseline_mape: float,
                threshold: float = 15.0) -> bool:
    """
    Returns True (and prints a warning) when MAPE has worsened
    beyond *threshold* % relative to the baseline.
    """
    if baseline_mape == 0:
        return False
    pct_change = ((current_mape - baseline_mape) / baseline_mape) * 100
    if pct_change > threshold:
        print(
            f"\n  DRIFT DETECTED — MAPE worsened by {pct_change:.1f}% "
            f"(threshold: {threshold}%). Retraining recommended.\n"
        )
        return True
    print(f" No drift. MAPE change: {pct_change:+.1f}% (threshold: ±{threshold}%)")
    return False
