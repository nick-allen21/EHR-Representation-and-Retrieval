from pathlib import Path

### CITATION: Used ChatGPT to help with BigQuery setup and configuration.
### We had no prior experience with the BigQuery Python client.

# bigquery stuff
import yaml
from google.cloud import bigquery
import pandas as pd


_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_client(config: dict | None = None) -> bigquery.Client:
    if config is None:
        config = load_config()
    return bigquery.Client(project=config["gcp_project_id"])


def run_query(sql: str, client: bigquery.Client | None = None,
              config: dict | None = None) -> pd.DataFrame:
    """Execute a Bigquery SQL string and return results as a DataFrame."""
    if config is None:
        config = load_config()
    if client is None:
        client = get_client(config)
    return client.query(sql).to_dataframe()


def build_id_filter(column: str, values: list, alias: str = "") -> str:
    """Build a SQL IN-clause filter for a list of integer IDs.

    Returns a string like: alias.column IN (1, 2, 3)
    """
    prefix = f"{alias}." if alias else ""
    ids = ", ".join(str(int(v)) for v in values)
    return f"{prefix}{column} IN ({ids})"
