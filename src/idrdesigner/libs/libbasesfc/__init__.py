"""A feature calculator for a base set of sequence features."""
import os
from pathlib import Path

default_config_path = Path(
    f"{os.path.dirname(os.path.abspath(__file__))}"
).joinpath("config.json")
