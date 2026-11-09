"""Steps 5 and 6: predict each VM's demand with a safety margin, then recommend sizes.

Demand is the 95th percentile of hourly maximum CPU percent over the next 7 days. Train and
calibration VMs learn from days 1-14 to predict days 15-21. Test VMs use days 8-21 to predict
days 22-28, so they are new VMs in a later week.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
LADDER = [2, 4, 8, 16, 24]
TARGET_USE = 0.70     # keep use at or below 70% of the new size
COVERAGE = 0.90
BANDS = [-1, 20, 40, 70, 101]   # bands of predicted CPU percent, each with its own margin
HOURS_PER_MONTH = 730
CATEGORY = {"Interactive": 0, "Delay-insensitive": 1, "Unknown": 2}
FEATURES = ["mean", "median", "p95", "peak", "std", "share_above_80", "weekend_weekday",
            "day_night", "week_change", "cores", "memory", "category_code"]


def size_for(demand_percent, current_cores, ladder=LADDER, target_use=TARGET_USE):
    """Smallest size that keeps demand at or below the target use. Never recommends a larger VM."""
    needed = demand_percent / 100 * current_cores / target_use
    return min([c for c in ladder if c >= needed and c <= current_cores], default=current_cores)


def risk_reasons(category, current_cores, new_cores, peak_percent, ladder=LADDER):
    """Why a person must approve this change. An empty list means low risk."""
    reasons = []
    if category != "Delay-insensitive":
        reasons.append(f"its workload category is {category.lower()}")
    if ladder.index(current_cores) - ladder.index(new_cores) > 1:
        reasons.append("it drops more than one size")
    if peak_percent / 100 * current_cores > new_cores:
        reasons.append("its recent peak was above the new size")
    return reasons


def days(usage, first, last):
    return usage[usage.hour.between((first - 1) * 24, last * 24 - 1)]


def calendar_groups(usage):
    """The trace doesn't name its weekdays, so find the quiet days and busy hours from days 1-14."""
    early = days(usage, 1, 14)
    by_day = early.groupby((early.hour // 24) % 7).max_cpu.mean()
    weekend = min(range(7), key=lambda d: by_day.get(d, 0) + by_day.get((d + 1) % 7, 0))
    by_hour = early.groupby(early.hour % 24).max_cpu.mean()
    return {weekend, (weekend + 1) % 7}, set(by_hour.nlargest(12).index)


def build_features(usage, first, vms, weekend, busy_hours):
    """One row per VM from the 14 days starting on day `first`."""
    window = days(usage, first, first + 13).copy()
    window["is_weekend"] = ((window.hour // 24) % 7).isin(weekend)
    window["is_busy"] = (window.hour % 24).isin(busy_hours)
    window["week"] = (window.hour // 24 - (first - 1)) // 7
    by_vm = window.groupby("vm_id").max_cpu
    out = pd.DataFrame({
        "mean": by_vm.mean(), "median": by_vm.median(), "p95": by_vm.quantile(0.95),
        "peak": by_vm.max(), "std": by_vm.std(),
        "share_above_80": window.assign(high=window.max_cpu > 80).groupby("vm_id").high.mean(),
    })
    ratio = lambda flag: (window[window[flag]].groupby("vm_id").max_cpu.mean()
                          / (window[~window[flag]].groupby("vm_id").max_cpu.mean() + 1))
    out["weekend_weekday"] = ratio("is_weekend")
    out["day_night"] = ratio("is_busy")
    by_week = window.groupby(["vm_id", "week"]).max_cpu.mean().unstack()
    out["week_change"] = by_week[1] - by_week[0]
    out = out.join(vms.set_index("vm_id")[["cores", "memory", "category"]])
    out["cores"] = out.cores.astype(int)
    out["memory"] = out.memory.str.replace(">64", "70").astype(int)
    out["category_code"] = out.category.map(CATEGORY)
    return out


def demand(usage, first, last):
    return days(usage, first, last).groupby("vm_id").max_cpu.quantile(0.95).rename("actual")


def conformal_margin(actual, predicted, coverage=COVERAGE):
    """Split conformal: the error level that held for the chosen share of calibration VMs."""
    errors = np.sort(np.asarray(actual) - np.asarray(predicted))
    rank = min(int(np.ceil((len(errors) + 1) * coverage)), len(errors)) - 1
    return float(errors[rank])


def band_margins(actual, predicted):
    """Quieter VMs have smaller errors, so one margin per band of predicted use is tighter
    where downsizing is possible, and still holds within each band."""
    bands = pd.cut(predicted, BANDS, labels=False)
    return {int(b): max(0.0, conformal_margin(actual[bands == b], predicted[bands == b])) for b in np.unique(bands)}


def fit_and_score(train_usage, test_usage, raw, vms, split, weekend, busy_hours):
    """Train on one version of the inputs, then return test predictions and summary numbers."""
    train_x = build_features(train_usage, 1, vms, weekend, busy_hours).join(demand(raw, 15, 21), how="inner")
    test_x = build_features(test_usage, 8, vms, weekend, busy_hours).join(demand(raw, 22, 28), how="inner")
    fit = train_x[train_x.index.isin(split["train"])]
    calibrate = train_x[train_x.index.isin(split["calibrate"])]
    test = test_x[test_x.index.isin(split["test"])].copy()

    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
    model.fit(fit[FEATURES], fit.actual)
    calibrated = model.predict(calibrate[FEATURES]).clip(0, 100)
    single = conformal_margin(calibrate.actual, calibrated)
    margins = band_margins(calibrate.actual.to_numpy(), calibrated)
    test["predicted"] = model.predict(test[FEATURES]).clip(0, 100)
    test["upper_single"] = (test.predicted + single).clip(0, 100)
    test["upper"] = (test.predicted + pd.cut(test.predicted, BANDS, labels=False).map(margins).fillna(single)).clip(0, 100)
    scores = {
        "calibration MAE": float(np.abs(calibrate.actual - calibrated).mean()),
        "test MAE": float(np.abs(test.actual - test.predicted).mean()),
        "single margin (CPU points)": single,
        "test coverage, single margin": float((test.actual <= test.upper_single).mean()),
        "test coverage": float((test.actual <= test.upper).mean()),
        "test VMs": len(test),
    }
    return test, scores, margins


def recommend(test, prices):
    rows = test.copy()
    rows["new_cores"] = [size_for(u, c) for u, c in zip(rows.upper, rows.cores)]
    rows["monthly_saving"] = (rows.cores.map(prices) - rows.new_cores.map(prices)) * HOURS_PER_MONTH
    rows["recommended"] = rows.new_cores < rows.cores
    rows["reasons"] = [
        "; ".join(risk_reasons(cat, c, n, p)) for cat, c, n, p in zip(rows.category, rows.cores, rows.new_cores, rows.peak)
    ]
    rows["risk"] = np.where(rows.reasons == "", "Low", "High")
    # CPU can't read above 100%, so these VMs may need more than the data can show. No size is suggested.
    rows["may_be_too_small"] = rows.share_near_100 > 0.25
    return rows


def main():
    RESULTS.mkdir(exist_ok=True)
    raw = pd.read_parquet(DATA / "cpu_hourly.parquet")
    vms = pd.read_parquet(DATA / "sample_vms.parquet")
    prices = pd.read_csv(DATA / "prices.csv").set_index("cores").hourly_price
    weekend, busy_hours = calendar_groups(raw)

    ids = np.random.default_rng(42).permutation(sorted(raw.vm_id.unique()))
    a, b = int(len(ids) * 0.70), int(len(ids) * 0.85)
    split = {"train": set(ids[:a]), "calibrate": set(ids[a:b]), "test": set(ids[b:])}

    versions = {
        "Raw inputs": (raw, raw),
        "Cleaned inputs": (pd.read_parquet(DATA / "usage_clean_train.parquet"),
                           pd.read_parquet(DATA / "usage_clean_test.parquet")),
    }
    results = {name: fit_and_score(tr, te, raw, vms, split, weekend, busy_hours) for name, (tr, te) in versions.items()}
    comparison = pd.DataFrame({name: r[1] for name, r in results.items()}).T
    comparison.round(3).to_csv(RESULTS / "raw_vs_cleaned.csv", index_label="inputs")
    print(comparison.round(3).to_string())

    # Chosen on calibration VMs, so the test week plays no part in the choice.
    chosen = comparison["calibration MAE"].idxmin()
    test, _, margins = results[chosen]
    labels = [f"{low + (low < 0)}-{min(high, 100)}%" for low, high in zip(BANDS, BANDS[1:])]
    table = pd.DataFrame({"predicted use": labels, "margin (CPU points)": [margins.get(i) for i in range(len(labels))]})
    table["test VMs"] = pd.cut(test.predicted, BANDS, labels=False).value_counts().reindex(range(len(labels))).values
    table["test coverage"] = (test.actual <= test.upper).groupby(pd.cut(test.predicted, BANDS, labels=False)).mean().reindex(range(len(labels))).values
    table.round(3).to_csv(RESULTS / "margins.csv", index=False)
    print(f"using {chosen}")
    print(table.round(3).to_string(index=False))

    recent = days(raw, 8, 21)
    test["share_near_100"] = recent.assign(near=recent.max_cpu >= 95).groupby("vm_id").near.mean().reindex(test.index)
    rows = recommend(test, prices)
    rows["upper_cores"] = rows.upper / 100 * rows.cores
    rows["peak_cores"] = rows.peak / 100 * rows.cores
    rows["inputs_used"] = chosen
    keep = ["category", "cores", "memory", "p95", "peak", "peak_cores", "predicted", "upper_single", "upper", "upper_cores", "actual",
            "new_cores", "monthly_saving", "recommended", "risk", "reasons", "may_be_too_small", "inputs_used"]
    rows[keep].round(4).to_csv(RESULTS / "vm_predictions.csv", index_label="vm_id")
    print(f"{rows.recommended.sum()} of {len(rows)} test VMs get a recommendation, "
          f"{(rows.recommended & (rows.risk == 'Low')).sum()} low risk, "
          f"${rows.monthly_saving.sum():,.0f} per month; {rows.may_be_too_small.sum()} flagged may be too small")


if __name__ == "__main__":
    main()
