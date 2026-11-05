"""Step 7: replay the held-out week (days 22-28) for the test VMs under four policies.

Savings and overload are always reported together, because the most reckless policy
will have the highest savings.
"""
from pathlib import Path

import pandas as pd

from right_size import days, size_for

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def replay(rows, week, new_cores, prices):
    """Savings and overload for one policy. `new_cores` is indexed by vm_id."""
    current_cost = rows.cores.map(prices)
    new_cost = new_cores.map(prices)
    hours = week[week.vm_id.isin(rows.index)].copy()
    hours["new_cores"] = hours.vm_id.map(new_cores)
    hours["cores"] = hours.vm_id.map(rows.cores)
    hours["overloaded"] = hours.max_cpu / 100 * hours.cores > hours.new_cores
    downsized = hours[hours.new_cores < hours.cores]
    return {
        "Savings (% of spend)": (current_cost - new_cost).sum() / current_cost.sum() * 100,
        "VMs downsized (%)": (new_cores < rows.cores).mean() * 100,
        "Overload (% of all VM-hours)": hours.overloaded.mean() * 100,
        "Overload (% of downsized VM-hours)": downsized.overloaded.mean() * 100 if len(downsized) else 0.0,
    }


def main():
    rows = pd.read_csv(RESULTS / "vm_predictions.csv", index_col="vm_id")
    usage = pd.read_parquet(ROOT / "data" / "cpu_hourly.parquet")
    prices = pd.read_csv(ROOT / "data" / "prices.csv").set_index("cores").hourly_price
    week = days(usage, 22, 28)
    recent_p95 = days(usage, 8, 21).groupby("vm_id").max_cpu.quantile(0.95).reindex(rows.index)

    sized = lambda values: pd.Series([size_for(v, c) for v, c in zip(values, rows.cores)], index=rows.index)
    policies = {
        "No change": rows.cores,
        "Simple rule (last 14 days)": sized(recent_p95),
        "ML only": sized(rows.predicted),
        "ML + margin": sized(rows.upper),
    }
    table = pd.DataFrame({name: replay(rows, week, new, prices) for name, new in policies.items()}).T
    table.round(2).to_csv(RESULTS / "policy_table.csv", index_label="policy")
    print(table.round(2).to_string())

    # Do the risk levels mean anything? Check them on what actually happened in the test week.
    by_risk = {}
    chosen = rows[rows.recommended]
    for risk, group in chosen.groupby("risk"):
        hours = week[week.vm_id.isin(group.index)]
        cores = hours.vm_id.map(group.cores)
        new = hours.vm_id.map(group.new_cores)
        by_risk[risk] = {
            "Recommendations": len(group),
            "Saving ($/month)": group.monthly_saving.sum(),
            "Share of test spend saved (%)": (group.cores.map(prices) - group.new_cores.map(prices)).sum()
            / rows.cores.map(prices).sum() * 100,
            "Overload (% of VM-hours)": (hours.max_cpu / 100 * cores > new).mean() * 100,
        }
    risk_table = pd.DataFrame(by_risk).T
    risk_table.round(2).to_csv(RESULTS / "risk_table.csv", index_label="risk")
    print(risk_table.round(2).to_string())


if __name__ == "__main__":
    main()
