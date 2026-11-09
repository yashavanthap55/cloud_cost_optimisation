"""Figures and bootstrap confidence intervals for the paper. Reads saved results and writes to paper/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from right_size import days, size_for

ROOT = Path(__file__).resolve().parent.parent
RESULTS, PAPER = ROOT / "results", ROOT / "paper"
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "savefig.dpi": 300})


def bootstrap(n_boot=2000):
    """Resample whole VMs, because hours from one VM are not independent."""
    rows = pd.read_csv(RESULTS / "vm_predictions.csv", index_col="vm_id")
    usage = pd.read_parquet(ROOT / "data" / "cpu_hourly.parquet")
    prices = pd.read_csv(ROOT / "data" / "prices.csv").set_index("cores").hourly_price
    week = days(usage, 22, 28)
    week = week[week.vm_id.isin(rows.index)]
    recent = days(usage, 8, 21).groupby("vm_id").max_cpu.quantile(0.95).reindex(rows.index)
    sized = lambda values: np.array([size_for(v, c) for v, c in zip(values, rows.cores)])
    policies = {"Simple rule": sized(recent), "ML only": sized(rows.predicted),
                "ML + single margin": sized(rows.upper_single), "ML + band margin": sized(rows.upper)}
    cores = rows.cores.to_numpy()
    cost = rows.cores.map(prices).to_numpy()
    picks = np.random.default_rng(42).integers(len(rows), size=(n_boot, len(rows)))
    position = pd.Series(range(len(rows)), index=rows.index)
    vm = week.vm_id.map(position).to_numpy()
    hours = np.bincount(vm, minlength=len(rows))
    draws, table = {}, []
    for name, new in policies.items():
        saving = cost - pd.Series(new).map(prices).to_numpy()
        over = np.bincount(vm, weights=week.max_cpu.to_numpy() / 100 * cores[vm] > new[vm], minlength=len(rows))
        small = new < cores
        draws[name] = dict(
            savings=saving[picks].sum(1) / cost[picks].sum(1) * 100,
            overload=over[picks].sum(1) / hours[picks].sum(1) * 100,
            conditional=(over * small)[picks].sum(1) / (hours * small)[picks].sum(1) * 100,
        )
        row = {"policy": name}
        for key, value in draws[name].items():
            row[f"{key}_low"], row[f"{key}_high"] = np.percentile(value, [2.5, 97.5])
        table.append(row)
    result = pd.DataFrame(table).set_index("policy")
    result.round(3).to_csv(RESULTS / "policy_ci.csv")
    print(result.round(2).to_string())
    for a, b, key in [("ML only", "Simple rule", "savings"), ("ML only", "Simple rule", "overload"),
                      ("ML + band margin", "Simple rule", "conditional"), ("ML + band margin", "ML only", "conditional")]:
        low, high = np.percentile(draws[a][key] - draws[b][key], [2.5, 97.5])
        print(f"{a} minus {b}, {key}: 95% CI [{low:.3f}, {high:.3f}]")
    hit = rows.actual <= rows.upper
    se = (hit.mean() * (1 - hit.mean()) / len(hit)) ** 0.5
    print(f"coverage {hit.mean():.3f}, 95% CI [{hit.mean() - 1.96 * se:.3f}, {hit.mean() + 1.96 * se:.3f}]")
    return result


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.set(xlim=(-0.1, 10.1), ylim=(-0.15, 10.45))
    ax.axis("off")

    def box(x, y, w, h, text, fill="#f2f2f0"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fill, ec=MUTED, lw=0.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color=INK, linespacing=1.15)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=MUTED, shrinkA=0, shrinkB=0))

    box(0, 9.1, 10, 1.2, "Azure VM trace (VM table, 5-minute CPU readings)\nand retail price list")
    box(0, 7.3, 10, 1.2, "Preparation: hourly fleet spend, hourly peak CPU,\nIsolation Forest cleaning of model inputs")
    box(0, 5.2, 4.4, 1.5, "Spend forecast\nXGBoost, 7 days ahead")
    box(5.6, 5.2, 4.4, 1.5, "VM demand model\nXGBoost, 95th percentile\nof hourly peak CPU")
    box(5.6, 3.5, 4.4, 1.1, "Split conformal margin\nper usage band")
    box(5.6, 1.8, 4.4, 1.1, "Sizing rule (70% target)\nand risk level")
    box(0, 0, 4.4, 1.2, "Replay on held-out\nweek (days 22-28)", "#ffffff")
    box(5.6, 0, 4.4, 1.2, "Explanation, dashboard\nand human approval", "#ffffff")
    arrow(5, 9.1, 5, 8.5)
    arrow(2.2, 7.3, 2.2, 6.7)
    arrow(7.8, 7.3, 7.8, 6.7)
    arrow(7.8, 5.2, 7.8, 4.6)
    arrow(7.8, 3.5, 7.8, 2.9)
    arrow(7.8, 1.8, 7.8, 1.2)
    arrow(5.6, 2.3, 2.2, 1.2)
    arrow(2.2, 5.2, 2.2, 1.2)
    fig.savefig(PAPER / "fig_pipeline.png", bbox_inches="tight", pad_inches=0.02)


def fig_forecast():
    df = pd.read_csv(RESULTS / "forecast.csv")
    day = df.hour / 24 + 1
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    ax.fill_between(day, df.low / 1000, df.high / 1000, color=BLUE, alpha=0.13, lw=0, label="Forecast band")
    ax.plot(day, df.forecast / 1000, color=BLUE, lw=1.1, label="XGBoost forecast")
    ax.plot(day, df.actual / 1000, color=ORANGE, lw=1.1, ls="--", label="Actual spend")
    ax.set(xlabel="Day of the trace", ylabel="Spend (thousand USD per hour)", xlim=(22, 29))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e3", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17), fontsize=8, handlelength=1.5, columnspacing=1)
    fig.savefig(PAPER / "fig_forecast.png", bbox_inches="tight", pad_inches=0.02)


def fig_tradeoff(ci):
    table = pd.read_csv(RESULTS / "policy_table.csv").set_index("Policy")
    names = {"Simple rule": "Simple rule (last 14 days)", "ML only": "ML only",
             "ML + single margin": "ML + single margin", "ML + band margin": "ML + margin by usage band"}
    offsets = {"Simple rule": (-5, -11, "right"), "ML only": (6, 4, "left"),
               "ML + single margin": (6, -9, "left"), "ML + band margin": (6, 3, "left")}
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    for short, full in names.items():
        x, y = table.loc[full, "Overload (% of all VM-hours)"], table.loc[full, "Savings (% of spend)"]
        c = ci.loc[short]
        ax.errorbar(x, y, xerr=[[max(0, x - c.overload_low)], [max(0, c.overload_high - x)]],
                    yerr=[[max(0, y - c.savings_low)], [max(0, c.savings_high - y)]],
                    fmt="o", ms=4, color=BLUE, ecolor=MUTED, elinewidth=0.6, capsize=1.5)
        dx, dy, ha = offsets[short]
        ax.annotate(short, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=8, color=INK)
    ax.set(xlabel="Overload (% of all VM-hours)", ylabel="Savings (% of spend)", xlim=(0, None), ylim=(0, None))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e6e6e3", lw=0.5)
    ax.set_axisbelow(True)
    fig.savefig(PAPER / "fig_tradeoff.png", bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    PAPER.mkdir(exist_ok=True)
    ci = bootstrap()
    fig_pipeline()
    fig_forecast()
    fig_tradeoff(ci)
