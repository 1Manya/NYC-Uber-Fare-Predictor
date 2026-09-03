"""
Model wrappers so anything saved via joblib exposes a plain .predict(X)
that returns fares in dollars — regardless of whether the underlying model
was trained on log1p(fare) or is an ensemble of two models. This means
app.py never needs to know or care about the log-transform / ensembling
detail; it just calls model.predict(X) as before.
"""
import numpy as np


class LogTargetRegressor:
    """Wraps a regressor that was trained on np.log1p(y). predict() inverts
    back to dollar units automatically with np.expm1().

    clip_log_range, if given, is a (min, max) bound applied to the
    *log-space* prediction before it gets inverted. This exists because an
    unregularized/linear model has no inherent ceiling on its output: a
    single unusual input can produce a log-prediction of 20+, and
    np.expm1(20) is roughly 485 million dollars. That's exactly what was
    tanking Linear Regression's R2 before (one bad row -> astronomical
    squared error). Tree models don't strictly need this (they can't
    predict outside the range of leaf values seen in training), but it's
    cheap and safe to apply to every model as a guardrail.
    """

    def __init__(self, inner_model, clip_log_range=None):
        self.inner_model = inner_model
        self.clip_log_range = clip_log_range

    def predict(self, X):
        log_pred = self.inner_model.predict(X)
        if self.clip_log_range is not None:
            lo, hi = self.clip_log_range
            log_pred = np.clip(log_pred, lo, hi)
        return np.expm1(log_pred)


class WeightedEnsembleRegressor:
    """Averages the (dollar-unit) predictions of two or more models. Each
    model in `models` must already expose predict() in dollar units — e.g.
    each one wrapped in a LogTargetRegressor first."""

    def __init__(self, models, weights):
        assert len(models) == len(weights)
        self.models = models
        self.weights = weights

    def predict(self, X):
        preds = [w * m.predict(X) for m, w in zip(self.models, self.weights)]
        return np.sum(preds, axis=0)
