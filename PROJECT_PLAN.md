# AI-Driven Cloud Cost Optimization: Project Plan

## What we are building

A Python system that uses public Azure VM data to:

1. Forecast next week's cloud spend.
2. Recommend smaller VMs where the current size is too large, with a safety margin on each recommendation.
3. Explain each recommendation in plain language and send risky ones to a person for approval.

Our literature review concludes that the missing piece is not a new forecasting algorithm. It is accurate predictions that a non-technical reviewer can understand and trust, so steps 5, 6 and 8 are where this project adds the most.

**Out of scope:** reinforcement learning, GPU workloads, multi-cloud, and changing real cloud resources. They need data or infrastructure we won't have, so list them as future work.

| Gap from the review | How we respond |
|---|---|
| Explanations for FinOps reviewers are an afterthought | Plain-language explanation for every recommendation (step 8) |
| Automation vs. control is rarely tested | Risk levels and human approval (step 6), overload measured (step 7) |
| Results are hard to reproduce | Public data and scripts anyone can rerun (steps 1 and 2) |
| Technical and business signals rarely meet | Partly: VM category feeds the risk level (step 6) |

## Tools

Python, pandas, pyarrow, scikit-learn, XGBoost and Streamlit. No cloud account is needed. The only possible cost is the optional LLM in step 8.

## Steps

### Step 1: Set up the project

Create a Git repository with this layout:

```
cloud-cost/
├── data/                 # not committed to Git
├── src/
│   ├── get_data.py
│   ├── prepare.py
│   ├── forecast_spend.py
│   ├── right_size.py
│   ├── evaluate.py
│   ├── explain.py
│   └── test_sizing.py
├── app.py
├── requirements.txt
└── README.md             # lists the scripts in run order
```

### Step 2: Get the data

**Usage data:** the [Azure Public Dataset V2](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md) covers 30 days and about 2.7 million VMs, with CPU readings every 5 minutes. It is 156 GB compressed across 198 files, so don't keep it all:

1. Download the VM table first. It has each VM's created and deleted times, category, core bucket and memory bucket.
2. Pick about 5,000 VMs that ran through at least day 28. Drop the largest, open-ended core bucket.
3. Download the CPU readings one file at a time. Keep only the rows for your VMs, reduce them to hourly values, save them as Parquet, and delete the raw file.

**Check first:** look at the timestamps in the first readings file. If the files are split by time, you must process all of them to get the full 30 days, so run this overnight.

**Prices:** the [Azure Retail Prices API](https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices) needs no login. Use this filter, changing the region if you prefer:

```
serviceName eq 'Virtual Machines' and armRegionName eq 'eastus' and priceType eq 'Consumption'
```

Remove the Spot, Low Priority and Windows rows. Choose one general-purpose VM family and map each core bucket to the size with the closest vCPU count.

**Done when:** `data/` contains `vm_table.parquet`, `cpu_hourly.parquet` and `prices.csv`.

### Step 3: Prepare the data

- **Spend series:** for each hour, add up the price of every VM running that hour. Use the full VM table, not the 5,000 VMs. A VM's price is fixed, so spend only changes when VMs start and stop.
- **Usage per VM:** for each VM and hour, keep the maximum CPU percent.
- **Anomalies:** use scikit-learn's `IsolationForest` to flag unusual hours for each VM. Make a cleaned copy with flagged hours replaced by that VM's median, and keep the raw copy too.

Only clean model inputs, never test targets. A real spike still hurts a downsized VM.

**Done when:** you have one hourly spend series and one hourly usage table in both raw and cleaned form.

### Step 4: Forecast spend

Predict hourly spend 7 days ahead and compare three models:

| Model | How it works |
|---|---|
| Linear projection | Straight-line trend fitted on days 1–21 (the baseline used in the review) |
| Seasonal naive | Same hour one week earlier |
| XGBoost | Hour, day of week, spend 168 hours ago, and the 24-hour average ending 168 hours ago |

Split by time and never shuffle. For XGBoost:

| Days | Use |
|---|---|
| 1–7 | History for lag features only |
| 8–17 | Train |
| 18–21 | Calibrate the forecast band |
| 22–28 | Test |

Don't give XGBoost anything newer than 168 hours before the hour it predicts. A forecast made a week ahead wouldn't have that data, and using it makes results look falsely good.

**Forecast band:** take the absolute errors on days 18–21. Their 90th percentile is the width of the band on each side of the forecast.

Report MAE in dollars, MAPE, and the share of test hours inside the band. If seasonal naive beats XGBoost, report that. It can happen with only 30 days and is still a valid result.

Skip LSTM and Temporal Fusion Transformer for now. The review notes they need a lot of data, and this series is only 30 days long.

**Done when:** you have a results table for all three models and a saved forecast with its band.

### Step 5: Predict each VM's demand, with a safety margin

Build one row per VM.

- **Target:** the 95th percentile of hourly maximum CPU percent over the next 7 days.
- **Features from the previous 14 days:** mean, median, 95th percentile, maximum and standard deviation of CPU percent; share of hours above 80%; weekday/weekend ratio; day/night ratio; week-over-week change; core bucket, memory bucket and VM category.

| Use | Feature days | Target days | VMs |
|---|---|---|---|
| Train | 1–14 | 15–21 | 70% |
| Calibrate | 1–14 | 15–21 | 15% |
| Test | 8–21 | 22–28 | 15% |

Then add the safety margin using split conformal prediction:

1. Train an XGBoost regressor on the training VMs.
2. On the calibration VMs, compute `actual - predicted`. The margin is the 90th percentile of these errors.
3. Set `upper = predicted + margin`, capped at 100. Actual demand should fall below this for about 90% of VMs.
4. Measure the real coverage on the test VMs and report it. The test week is later than the calibration week, so it may fall short of 90%.

Run this with raw and cleaned inputs and report both. That tests the review's claim that clean data matters as much as the model.

**Done when:** every test VM has `predicted` and `upper` values, and you have measured the coverage.

### Step 6: Turn predictions into recommendations

Apply the same sizing rule to every VM:

```python
upper_cores = upper / 100 * current_cores
needed = upper_cores / 0.70     # keep use at or below 70% of the new size
new_cores = smallest core bucket >= needed
recommend only if new_cores < current_cores
monthly_saving = (current_price - new_price) * 730
```

The 70% target is our own choice, so state it in the report.

Give each recommendation a risk level. Check the exact category labels in the VM table first.

| Risk | When | Action |
|---|---|---|
| High | Category is interactive or unknown, **or** it drops more than one size, **or** the peak in the last 14 days is above the new size | A person must approve it |
| Low | Everything else | Can be approved in bulk |

Flag VMs that often run near 100% as "may be too small", but don't suggest a size. CPU can't read above 100%, so the data can't show how much more they need.

Write `test_sizing.py` with a few `assert` checks, since this rule decides the savings: it never recommends a larger size, never picks a size below `needed`, and makes no recommendation for a VM already at the smallest size.

**Done when:** you have a table of recommendations with savings and risk levels, and the test passes.

### Step 7: Test against the held-out week

Replay days 22–28 for the test VMs under four policies:

| Policy | Value used in the sizing rule |
|---|---|
| No change | None, VMs keep their size |
| Simple rule | 95th percentile from the last 14 days |
| ML only | `predicted` |
| ML + margin | `upper` |

Measure these for each policy:

- **Savings:** percent of the test VMs' spend.
- **VMs downsized:** percent of test VMs.
- **Overload:** percent of VM-hours where `hourly max CPU / 100 * current_cores > new_cores`.

This table is the project's main result. Always show savings and overload together, because the most reckless policy will have the highest savings. You can compare with the roughly 18% savings in the review, but note that those studies used different data.

**Done when:** the table is filled in and you can say what the safety margin costs in savings and gains in overload.

### Step 8: Explain each recommendation

Start with a template. It needs no API and can't invent numbers. For example, with made-up numbers:

> **VM a1b2: move from 8 to 4 vCPUs and save about $140 a month.** Its highest use in the last two weeks was 3.1 vCPUs. We expect busy-hour use next week to stay under 2.6 vCPUs, which leaves 35% spare capacity. Estimates like this held about 90% of the time in testing. **Risk: low.**

Use your measured coverage from step 5 in place of the 90%.

**Optional LLM layer:** give an LLM the same facts and ask it to reword them for a finance reader. It should only reword, never calculate. Reject any output containing a number that isn't in the facts, and fall back to the template.

**Small user check:** ask 3–5 people who aren't engineers to read 10 recommendations. Record whether they understood each one and whether they would approve it. The review notes this is rarely studied, so even a small test adds something.

**Done when:** every recommendation has an explanation and you have the user check results.

### Step 9: Build the dashboard

Create one Streamlit page in `app.py` that reads the saved results:

- Spend forecast chart with its band.
- Total possible savings, and the number of low-risk and high-risk recommendations.
- Recommendation table with VM, current size, new size, saving and risk.
- Explanation and Approve/Reject buttons for the selected VM, with decisions saved to `data/decisions.csv`.

Approvals are only logged. Nothing changes real cloud resources.

**Done when:** you can go from the forecast to an approved recommendation on the page.

### Step 10: Write the report and prepare the demo

**Report results:** forecast table, coverage, the policy table, raw vs. cleaned inputs, the user check, and the gaps table from the top of this plan.

**State these limitations plainly:**

- **One provider and region:** the data is first-party Azure workload from one region, so results may not generalize.
- **Short period:** 30 days can't show monthly cycles or long-term drift.
- **No memory usage:** the data only has a memory size bucket, so we assume memory can shrink with cores. This makes savings optimistic.
- **Capped CPU:** usage can't show demand above a VM's current size, so we only evaluate downsizing.
- **Estimated costs:** these are current retail prices applied to an older trace, without discounts, so they aren't real bills.
- **Long-running VMs only:** right-sizing results exclude VMs that ran for less than 28 days.

**Demo (5 minutes):** the problem, the forecast, one recommendation and its explanation, an approval, then the policy table.

## Suggested schedule

This assumes about four weeks, so compress it to fit your deadline.

| Week | Steps | Owner |
|---|---|---|
| 1 | 1–3: setup, data, preparation | Data and forecasting |
| 2 | 4–5: spend forecast, VM demand model | Data and forecasting, right-sizing |
| 3 | 6–7: recommendations, evaluation | Right-sizing |
| 4 | 8–10: explanations, dashboard, report | Explanations and report, with everyone on the final report |

Start the step 2 download early. It is the step most likely to take longer than expected.
