"""Shared model plots: per-model diagnostics and the cross-model comparison."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import LearningCurveDisplay

from news_category.evaluate import SCORER, make_folds


def plot_confusion_matrix(y_true, y_pred, labels: list, ax=None) -> None:
  """Row-normalized confusion matrix, so each true class reads as a rate regardless of its size."""
  ax = ax or plt.gca()
  matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
  sns.heatmap(matrix, ax=ax, xticklabels=labels, yticklabels=labels, cmap="Blues", vmin=0, vmax=1, cbar=False)
  ax.set_title("Confusion Matrix")
  ax.set_xlabel("Predicted")
  ax.set_ylabel("True")


def plot_per_class_f1(y_true, y_pred, labels: list, ax=None) -> None:
  """Per-class F1 as a sorted bar chart, exposing which categories the model handles worst."""
  ax = ax or plt.gca()
  scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
  sr_f1 = pd.Series(scores, index=labels).sort_values()
  sr_f1.plot.barh(ax=ax, title="Per-Class F1", xlabel="F1", xlim=(0, 1))


def plot_learning_curve(estimator, X, y, ax=None) -> None:
  """Macro-F1 against training-set size, with train and validation curves to reveal over/underfitting."""
  ax = ax or plt.gca()
  LearningCurveDisplay.from_estimator(
    estimator, X, y, cv=make_folds(), scoring=SCORER, score_name="Macro F1", train_sizes=np.linspace(0.1, 1.0, 5), ax=ax
  )
  ax.set_title("Learning Curve")


def plot_metric_comparison(results: dict, ax=None) -> None:
  """Grouped bars per metric, one bar per config, labeled with exact scores, for picking the winner."""
  ax = ax or plt.gca()
  df_results = pd.DataFrame(results)  # metrics as rows (x), configs as columns (grouped bars)
  df_results.plot.bar(ax=ax, title="Metric Comparison", xlabel="", ylim=(0, 1), rot=0)
  for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", rotation=90, padding=2, fontsize=7)
  ax.legend(loc="lower right", ncol=2)
