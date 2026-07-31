"""Shared console/notebook output helpers."""

from IPython.display import display


def show(name: str, value) -> None:
  """Print a labeled value, rendering DataFrames as tables."""
  print(f"── {name} ".ljust(44, "─"))
  if hasattr(value, "_repr_html_"):
    display(value)
  else:
    print(value)
