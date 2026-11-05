"""Step 3: build the hourly spend series and the cleaned usage tables.

Writes data/spend_hourly.parquet and one cleaned usage table per feature window.
Only model inputs get cleaned. Targets always use raw usage, because a real spike still
hurts a downsized VM.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import IsolationForest

DATA = Path(__file__).resolve().parent.parent / "data"
BIN = 300                      # the trace records times in 5-minute steps
BINS = 30 * 24 * 12            # 30 days
WINDOWS = {"train": (1, 14), "test": (8, 21)}   # feature windows from step 5, in days


def build_spend():
    """Hourly spend across the full VM table, charging each VM only for the time it ran."""
    vms = pd.read_parquet(DATA / "vm_table.parquet", columns=["created", "deleted", "cores"])
    prices = pd.read_csv(DATA / "prices.csv")
    price = dict(zip(prices.cores.astype(str), prices.hourly_price))
    vms = vms[vms.cores.isin(list(price))]           # ">24" has no known size, so it can't be priced
    rate = vms.cores.map(price).to_numpy(float)
    change = np.zeros(BINS + 1)
    np.add.at(change, np.clip(vms.created.to_numpy() // BIN, 0, BINS - 1), rate)
    np.add.at(change, np.clip(vms.deleted.to_numpy() // BIN + 1, 0, BINS), -rate)
    dollars_per_hour = np.cumsum(change)[:BINS]
    spend = dollars_per_hour.reshape(-1, 12).mean(axis=1)
    pd.DataFrame({"hour": np.arange(len(spend)), "spend": spend}).to_parquet(DATA / "spend_hourly.parquet", index=False)
    print(f"spend: {len(vms):,} VMs, ${spend.min():,.0f}-${spend.max():,.0f} per hour")


def clean_vm(vm):
    """Replace hours flagged as unusual with this VM's median."""
    values = vm.max_cpu.to_numpy(float)
    if len(values) < 10:
        return vm.assign(is_anomaly=False)
    flagged = IsolationForest(n_estimators=30, contamination=0.02, random_state=42).fit_predict(values[:, None]) == -1
    return vm.assign(max_cpu=np.where(flagged, np.median(values), values), is_anomaly=flagged)


def build_clean_usage():
    usage = pd.read_parquet(DATA / "cpu_hourly.parquet")
    for name, (first, last) in WINDOWS.items():
        # Each window is cleaned using only its own days, so nothing later leaks in.
        window = usage[usage.hour.between((first - 1) * 24, last * 24 - 1)]
        groups = [vm for _, vm in window.groupby("vm_id")]
        cleaned = pd.concat(Parallel(n_jobs=4, batch_size=50)(delayed(clean_vm)(vm) for vm in groups))
        cleaned.to_parquet(DATA / f"usage_clean_{name}.parquet", index=False)
        print(f"cleaned {name} window: {cleaned.is_anomaly.mean():.1%} of hours flagged")


if __name__ == "__main__":
    build_spend()
    build_clean_usage()
