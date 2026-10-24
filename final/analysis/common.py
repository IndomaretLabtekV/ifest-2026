"""Shared paths + the raw-string parser used by every analysis script."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WORK = ROOT / "work"
FIG = ROOT / "outputs" / "figs"
WORK.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

LAG = range(12)
FLOWS = ["q_bf_inc_m3", "q_qf_inc_m3", "q_ws_cum_m3", "q_ir_wd_inc_m3", "q_th_wd_inc_m3", "q_ps_wd_inc_m3"]
LAG_COLS = [f"{f}_lag{k}" for f in FLOWS for k in LAG]
STATIC = ["inc_aream2", "cum_aream2", "pop_dec_2000", "pop_dec_2010", "pop_dec_2020", "pop_acs_2020"]
CLIM = ["clim_median_now", "clim_median_next"]
NUM_COLS = STATIC + LAG_COLS + CLIM

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}


def parse_num(s: pd.Series) -> pd.Series:
    """Raw string -> float.
    '1,234.567' -> US thousands; '1234,567' -> decimal comma (always 3 decimals).
    A lone comma with no dot is a decimal comma (verified against aux_twin columns in 01)."""
    s = s.astype("string").str.strip()
    both = s.str.contains(",", regex=False) & s.str.contains(".", regex=False)
    s = s.mask(both, s.str.replace(",", "", regex=False))
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce").astype("float64")


def parse_month(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.lower()
    num = pd.to_numeric(s, errors="coerce")
    name = s.map(lambda x: next((v for k, v in MONTHS.items() if isinstance(x, str) and k.startswith(x[:3])), np.nan))
    return num.fillna(name).astype("int64")


def load_raw():
    tr = pd.read_csv(DATA / "train.csv", dtype=str).rename(columns={"Unnamed: 0": "row_id"})
    te = pd.read_csv(DATA / "test_public.csv", dtype=str)
    return tr, te


def load_clean():
    """Cached parsed frames written by 01_data_quality.py."""
    return pd.read_parquet(WORK / "train.parquet"), pd.read_parquet(WORK / "test.parquet")


if __name__ == "__main__":
    t = parse_num(pd.Series(["1,234.5", "1234,567", "3,501.529", "-9,999", "1.03705e+06", None, "0"]))
    assert np.allclose(t.fillna(-7).tolist(), [1234.5, 1234.567, 3501.529, -9.999, 1037050, -7, 0])
    assert parse_month(pd.Series(["1", "Feb", "february", "May", "12"])).tolist() == [1, 2, 2, 5, 12]
    print("ok")
