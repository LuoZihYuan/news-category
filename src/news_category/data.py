"""Shared data access: versioned splits and field assembly."""

from pathlib import Path
from itertools import combinations
import json

import pandas as pd

DIR_PROCESSED = Path("data/processed")

# The field configurations under test: every non-empty combination of the text
# columns. Each name is also its spec, the columns recovered by splitting on "+",
# so the string doubles as a result label.
FIELDS = ["headline", "short_description", "authors"]
CONFIGS = ["+".join(combo) for size in range(1, len(FIELDS) + 1) for combo in combinations(FIELDS, size)]


def load_split(version: str) -> tuple:
  """Load a version's train/test frames and its canonical label order.

  Returns ((df_train, df_test), labels) for data/processed/<version>/.
  """
  directory = DIR_PROCESSED / version
  df_train = pd.read_parquet(directory / "train.parquet")
  df_test = pd.read_parquet(directory / "test.parquet")
  with open(directory / "labels.json") as f:
    labels = json.load(f)
  return (df_train, df_test), labels


def assemble_fields(df: pd.DataFrame, config: str) -> pd.Series:
  """Join the config's columns into one input string per row, split on '+'.

  The separator is DistilBERT's `[SEP]`; for TF-IDF it is just a token boundary.
  """
  fields = config.split("+")
  return df[fields].agg(" [SEP] ".join, axis=1)
