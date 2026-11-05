"""Step 9: one-page dashboard. It reads saved results and only logs decisions.

Run: streamlit run app.py
"""
import csv
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DECISIONS = ROOT / "data" / "decisions.csv"
FORECAST_COLOR = "#2a78d6"
ACTUAL_COLOR = "#eb6834"


@st.cache_data
def load(name, **options):
    return pd.read_csv(RESULTS / name, **options)


def read_decisions():
    """Each VM's latest decision."""
    if not DECISIONS.exists():
        return {}
    log = pd.read_csv(DECISIONS).drop_duplicates("vm_id", keep="last")
    return dict(zip(log.vm_id, log.decision))


def save_decisions(rows, decision):
    DECISIONS.parent.mkdir(exist_ok=True)
    is_new = not DECISIONS.exists()
    with open(DECISIONS, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["time", "vm_id", "decision", "risk", "monthly_saving"])
        now = datetime.now().isoformat(timespec="seconds")
        for row in rows.itertuples():
            writer.writerow([now, row.vm_id, decision, row.risk, row.monthly_saving])


def spend_chart(forecast):
    df = forecast.assign(day=forecast.hour / 24 + 1)
    x = alt.X("day:Q", title="Day of the trace", scale=alt.Scale(domain=[22, 29], nice=False),
              axis=alt.Axis(values=list(range(22, 30)), format="d", grid=False))
    y_scale = alt.Scale(zero=False)
    band = alt.Chart(df).mark_area(opacity=0.15, color=FORECAST_COLOR).encode(
        x=x, y=alt.Y("low:Q", title="Spend ($ per hour)", scale=y_scale, axis=alt.Axis(format="$,.0f")), y2="high:Q")
    lines = alt.Chart(df).transform_fold(["forecast", "actual"], as_=["series", "value"]).transform_calculate(
        label="datum.series === 'forecast' ? 'Forecast' : 'Actual'"
    ).mark_line(strokeWidth=2).encode(
        x=x, y=alt.Y("value:Q", scale=y_scale),
        color=alt.Color("label:N", scale=alt.Scale(domain=["Forecast", "Actual"], range=[FORECAST_COLOR, ACTUAL_COLOR]),
                        legend=alt.Legend(orient="top", title=None)),
    )
    hover = alt.selection_point(fields=["day"], nearest=True, on="pointermove", empty=False)
    crosshair = alt.Chart(df).mark_rule(color="#52514e", strokeWidth=1).encode(
        x=x,
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
        tooltip=[alt.Tooltip("day:Q", title="Day", format=".1f"),
                 alt.Tooltip("forecast:Q", title="Forecast", format="$,.0f"),
                 alt.Tooltip("actual:Q", title="Actual", format="$,.0f"),
                 alt.Tooltip("low:Q", title="Band low", format="$,.0f"),
                 alt.Tooltip("high:Q", title="Band high", format="$,.0f")],
    ).add_params(hover)
    return (band + lines + crosshair).properties(height=320)


st.set_page_config(page_title="Cloud Cost Optimizer", page_icon=":material/savings:", layout="wide")
st.title("Cloud Cost Optimizer")
st.caption("Real Azure VM trace, seen from the end of day 21 and checked against days 22–28. "
           "Costs are estimates from retail prices. Approvals are only logged, so nothing changes real cloud resources.")

forecast = load("forecast.csv")
band = load("forecast_band.csv", index_col="item").value
metrics = load("forecast_metrics.csv")
recs = load("recommendations.csv", keep_default_na=False)
predictions = load("vm_predictions.csv")
policies = load("policy_table.csv")
risks = load("risk_table.csv")
comparison = load("raw_vs_cleaned.csv", index_col="inputs")
used = predictions.inputs_used.iloc[0]
coverage = float(comparison.loc[used, "test coverage"])
decisions = read_decisions()

with st.container(horizontal=True):
    st.metric("Forecast spend, next 7 days", f"${forecast.forecast.sum():,.0f}", border=True,
              help="All priced VMs in the trace, using today's retail prices.")
    st.metric("Possible saving", f"${recs.monthly_saving.sum():,.0f}/month", border=True,
              help=f"Across the {len(predictions):,} held-out test VMs, if every recommendation is approved.")
    st.metric("Low-risk recommendations", f"{(recs.risk == 'Low').sum():,}", border=True)
    st.metric("Need a person's approval", f"{(recs.risk == 'High').sum():,}", border=True)
    st.metric("Safety margin held", f"{coverage:.0%}", border=True,
              help="Share of test VMs whose real demand stayed under the estimate. The target was 90%.")

with st.container(border=True):
    st.subheader("Spend forecast")
    st.caption(f"Model: {band['model']}. Shaded area: forecast band. "
               f"{float(band['test_hours_inside_band']):.0%} of test hours fell inside it (target 90%).")
    st.altair_chart(spend_chart(forecast), width="stretch",
                    alt="Hourly spend forecast against actual spend for days 22 to 28, with a shaded band.")

st.subheader("Recommendations")
left, right = st.columns([3, 2], gap="medium")

with left:
    choice = st.segmented_control("Show", ["All", "Low", "High"], default="All", key="risk_filter")
    view = recs if choice in (None, "All") else recs[recs.risk == choice]
    view = view.assign(status=[decisions.get(v, "Pending") for v in view.vm_id])
    table = view[["vm_short", "category", "cores", "new_cores", "monthly_saving", "risk", "status"]]
    event = st.dataframe(
        table, hide_index=True, on_select="rerun", selection_mode="single-row", height=380, key=f"recs_{choice}",
        alt="Recommended VM size changes with saving, risk and approval status",
        column_config={
            "vm_short": "VM", "category": "Workload",
            "cores": st.column_config.NumberColumn("Now (vCPU)", format="%d"),
            "new_cores": st.column_config.NumberColumn("New (vCPU)", format="%d"),
            "monthly_saving": st.column_config.NumberColumn("Saving/month", format="$%.0f"),
            "risk": "Risk", "status": "Status",
        },
    )
    low_pending = recs[(recs.risk == "Low") & (~recs.vm_id.isin(decisions))]
    st.button(f"Approve all {len(low_pending)} pending low-risk", icon=":material/done_all:",
              disabled=low_pending.empty, on_click=save_decisions, args=(low_pending, "Approved"))

with right:
    with st.container(border=True):
        if view.empty:
            st.info("No recommendations match this filter.")
        else:
            picked = view.iloc[event.selection.rows[0] if event.selection.rows else 0]
            st.markdown(f"**VM {picked.vm_short}** · {picked.risk} risk · {picked.status}")
            st.write(picked.explanation)
            row = view[view.vm_id == picked.vm_id]
            with st.container(horizontal=True):
                st.button("Approve", icon=":material/check:", type="primary", key="approve",
                          on_click=save_decisions, args=(row, "Approved"))
                st.button("Reject", icon=":material/close:", key="reject",
                          on_click=save_decisions, args=(row, "Rejected"))
    too_small = int(predictions.may_be_too_small.sum())
    st.caption(f"{too_small} other test VMs often run near 100% and may be too small. No size is suggested for them, "
               "because CPU can't read above 100%, so the data can't show how much more they need.")

st.subheader("Did it work?")
st.caption("Replay of the held-out week. Overload means the VM's hourly peak would have been above its new size.")
with st.container(border=True):
    st.markdown("**Savings and overload by policy**")
    st.dataframe(policies, hide_index=True, alt="Savings and overload for four sizing policies",
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in policies.columns[1:]})
col_a, col_b = st.columns(2, gap="medium")
with col_a.container(border=True):
    st.markdown("**Do the risk levels mean anything?**")
    st.dataframe(risks, hide_index=True, alt="Recommendations, saving and overload by risk level")
with col_b.container(border=True):
    st.markdown("**Forecast accuracy, days 22–28**")
    st.dataframe(metrics, hide_index=True, alt="Forecast error for three models")
