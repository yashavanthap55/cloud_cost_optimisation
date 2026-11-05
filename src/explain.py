"""Step 8: write a plain-language explanation for every recommendation.

A template needs no API and can't invent numbers. Also writes results/user_check.csv, a sheet
for non-engineers to fill in by hand.
"""
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"


def explain(row, coverage):
    spare = 1 - row.upper_cores / row.new_cores
    text = (
        f"VM {row.vm_short}: move from {row.cores} to {row.new_cores} vCPUs and save about "
        f"${row.monthly_saving:,.0f} a month. Its highest use in the last two weeks was {row.peak_cores:.1f} vCPUs. "
        f"We expect busy-hour use next week to stay under {row.upper_cores:.1f} vCPUs, which leaves "
        f"{spare:.0%} spare capacity. Estimates like this held about {coverage:.0%} of the time in testing. "
    )
    if row.risk == "Low":
        return text + "Risk: low."
    return text + f"Risk: high, because {row.reasons}. A person must approve this change."


def main():
    rows = pd.read_csv(RESULTS / "vm_predictions.csv")
    rows = rows[rows.recommended].copy()
    rows["reasons"] = rows.reasons.fillna("")

    comparison = pd.read_csv(RESULTS / "raw_vs_cleaned.csv", index_col="inputs")
    coverage = comparison.loc[rows.inputs_used.iloc[0], "test coverage"]

    rows["vm_short"] = rows.vm_id.str[:8]
    rows["explanation"] = [explain(row, coverage) for row in rows.itertuples()]
    columns = ["vm_id", "vm_short", "category", "cores", "new_cores", "monthly_saving", "risk", "reasons", "explanation"]
    rows = rows.sort_values("monthly_saving", ascending=False)
    rows[columns].to_csv(RESULTS / "recommendations.csv", index=False)
    print(f"{len(rows)} explanations written")

    # Up to 5 low-risk and 5 high-risk examples. The two answer columns are left blank for real people.
    picks = pd.concat(group.sample(min(5, len(group)), random_state=42) for _, group in rows.groupby("risk"))
    sheet = picks[["vm_short", "explanation"]].assign(understood_yes_no="", would_approve_yes_no="", reader_name="")
    sheet.to_csv(RESULTS / "user_check.csv", index=False)
    print(rows.explanation.iloc[0])


if __name__ == "__main__":
    main()
