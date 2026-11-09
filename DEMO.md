# Demo Script (5 minutes)

**Start the dashboard** from the project folder, before you begin:

```
.venv\Scripts\streamlit run app.py
```

It opens at http://localhost:8501 and works offline, because it reads the saved results.

| Time | Show | Say |
|---|---|---|
| 0:30 | Top of the page | Cloud spend is hard to control because teams over-provision VMs. Our literature review found the missing piece isn't a new algorithm. It's predictions a non-technical reviewer can trust. |
| 1:00 | Spend forecast chart | This uses a real Azure trace of 2.7 million VMs. From the end of day 21, we forecast the next week. XGBoost's error was 0.51%, against 1.26% for a straight-line projection. |
| 1:30 | Recommendations table and explanation | For each VM we predict busy-hour demand, add a safety margin, and pick the smallest size that stays under 70% use. Every recommendation explains itself in plain language. |
| 0:30 | Click **Approve**, then **Approve all low-risk** | Low-risk changes can be approved in bulk. High-risk ones need a person. Decisions are only logged, so nothing changes real VMs. |
| 1:00 | "Did it work?" tables | We replayed a week the model never saw. With the margin we saved 4.33% of spend and overloaded 0.04% of VM-hours. The simple rule saved 8.20% with 0.07% overload, so the margin trades savings for caution. |
| 0:30 | Finish | Our main contribution is a full, reproducible pipeline on public data, with safety margins, explanations and human approval, tested honestly on held-out data. |

## Questions you may be asked

**Is the data real?**
Yes. Usage comes from the Azure Public Dataset V2, and prices from Azure's retail price API. Costs are estimates, because we apply today's Dsv5 Linux prices to a 2019 trace.

**Did ML beat the simple rule?**
Not clearly. ML alone saved 8.51% with 0.08% overload, and the simple rule saved 8.20% with 0.07%. Recent usage already predicts next week's usage well. What ML adds here is a calibrated margin and a way to measure risk, not much extra saving.

**What does the safety margin actually do?**
It roughly halves both savings and total overload, because fewer VMs get changed. Among VMs that are downsized, the overload rate stays about the same (0.94% against 0.87% for the simple rule). So it makes the system more cautious, but it doesn't pick safer VMs.

**Did the margin hit its 90% target?**
Nearly. Real demand stayed under the estimate for 87.8% of test VMs. The shortfall is expected, because the test week is later than the calibration week.

**Why is the saving lower than the roughly 18% in the literature?**
78% of test VMs are already at the smallest size (2 vCPUs), and they make up 57% of spend, so they can't shrink further. Our margin is also deliberately cautious, and the data is different.

**Do the risk levels predict overload?**
No, and we report that. Low-risk VMs overloaded 1.22% of hours, and high-risk VMs 0.36%. Many high-risk VMs are flagged because they are interactive, so the label reflects how costly a mistake would be more than how likely one is.

**Isn't a 59% forecast improvement huge?**
It sounds bigger than it is. Fleet spend is very stable, so even the straight line is only 1.26% off. The forecast band is also too wide: 100% of test hours fell inside it, against a 90% target.

**Did cleaning the data help?**
Barely. Test error was 9.19 CPU points with cleaning and 9.21 without. That doesn't support the review's claim that clean data matters as much as the model, at least on this data. We kept raw inputs because they did better on calibration data.

**Why not LSTM or reinforcement learning?**
There are only 30 days of data, and the review notes deep models need much more. Reinforcement learning would need a live system to train on.

## Limits to state if asked

- **Sampled hours:** we used 99 sampled hours out of the 672 in 28 days, so peaks are likely underestimated. We read one hour from each of 100 of the 184 files, which covers the month evenly.
- **VM sample:** 20,000 VMs that ran for all 28 days, so short-lived VMs are excluded from right-sizing.
- **No memory usage data:** we assume memory can shrink with cores, which makes savings optimistic.
- **Capped CPU:** usage can't show demand above a VM's current size, so we only evaluate downsizing.
- **One provider and region:** results may not generalize.

## Not done yet

- **User check:** `results/user_check.csv` has 10 explanations ready, but real non-engineers still need to read and answer them. Don't claim this was tested.
- **LLM explanations:** we skipped this optional step. The template explanations can't invent numbers and need no API.
- **Written report:** the IEEE paper is in `paper/`.

## Changes from the plan

- **Sampling:** we read 1 hour per file from 100 files instead of every reading, because the download was too slow for the deadline.
- **More VMs:** we used 20,000 instead of 5,000, which cost no extra download, because most VMs can't shrink.
- **Margins by usage band:** we added one margin per band of predicted use, because a single margin was too blunt. Both versions are in the results table.
- **Prices:** Dsv5's per-vCPU rate prices the 24-vCPU bucket, which has no exact Dsv5 size. VMs can also move to 16 vCPUs.
