"""Shared model tuning and scoring, fixed so every model is compared on identical folds and metrics."""

from contextlib import contextmanager
from typing import NamedTuple
import os
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

RANDOM_STATE = 42
N_SPLITS = 5

# Macro averaging weights every class equally, so minority categories count as much
# as POLITICS; this is the comparison metric the whole field study turns on.
SCORER = make_scorer(f1_score, average="macro")


@contextmanager
def suppress_gridsearch_warnings():
  """Silence the convergence and fit-failure noise a wide grid search produces."""
  original = os.environ.get("PYTHONWARNINGS", "")
  os.environ["PYTHONWARNINGS"] = "ignore"
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FitFailedWarning)
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    try:
      yield
    finally:
      os.environ["PYTHONWARNINGS"] = original


def make_folds() -> StratifiedKFold:
  """The one cross-validation splitter, fixed so classic and deep models share identical folds."""
  return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
  """Macro-averaged scores for one set of predictions, comparable across models."""
  return {
    "f1": f1_score(y_true, y_pred, average="macro"),
    "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
    "recall": recall_score(y_true, y_pred, average="macro"),
    "accuracy": accuracy_score(y_true, y_pred),
  }


class TuningResult(NamedTuple):
  """The fitted best model, its test predictions, its test scores, and its winning hyperparameters."""

  estimator: ClassifierMixin
  predictions: np.ndarray
  scores: dict
  best_params: dict


def tune_hyperparameter(
  clf: ClassifierMixin,
  grid: dict,
  X_train,
  y_train,
  X_test,
  y_test,
  n_jobs: int | None = -1,
) -> TuningResult:
  """Grid-search a classic model over the shared folds, then score on the test split.

  Returns the fitted best model, its test predictions, and its scores, so the caller
  can plot from them without re-running the search. Search progress is always logged;
  callers are expected to clear it (the notebooks do, per configuration).
  """
  search = GridSearchCV(clf, grid, scoring=SCORER, cv=make_folds(), n_jobs=n_jobs, verbose=10, return_train_score=True)
  with suppress_gridsearch_warnings():
    search.fit(X_train, y_train)
  best = search.best_estimator_
  y_pred = best.predict(X_test)
  return TuningResult(best, y_pred, evaluate(y_test, y_pred), search.best_params_)
