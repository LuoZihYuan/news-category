"""Shared console/notebook output helpers."""

from IPython.display import display


def show(name: str, value=None) -> None:
  """Print a labeled value, rendering DataFrames as tables; with no value, print just the label."""
  print(f"── {name} ".ljust(44, "─"))
  if value is None:
    return
  if hasattr(value, "_repr_html_"):
    display(value)
  else:
    print(value)
