# Cloud Cost Optimizer

Forecasts cloud spend and recommends smaller VMs using a real Azure VM trace. Each recommendation has a safety margin and a plain-language explanation, and risky ones wait for a person's approval. Nothing changes real cloud resources.

See [REPORT.md](REPORT.md) for results, [DEMO.md](DEMO.md) for the demo script, and [PROJECT_PLAN.md](PROJECT_PLAN.md) for the original plan.

## Show the dashboard

The results are already saved in `results/`, so this needs no download:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Rerun everything

Run these in order. `get_data.py` is the slow one: it streams about 40 GB and can be stopped and restarted without losing finished files.

| Script | Plan step | What it does |
|---|---|---|
| `python src/get_data.py` | 2 | Downloads the VM table and prices, picks 20,000 VMs, and reduces CPU readings to hourly values |
| `python src/prepare.py` | 3 | Builds the hourly spend series and cleans usage with `IsolationForest` |
| `python src/forecast_spend.py` | 4 | Compares three spend forecasts and calibrates the band |
| `python src/right_size.py` | 5–6 | Predicts each VM's demand, adds the safety margin, and makes recommendations |
| `python src/test_sizing.py` | 6 | Checks the sizing and risk rules |
| `python src/evaluate.py` | 7 | Replays the held-out week under four policies |
| `python src/explain.py` | 8 | Writes explanations and the user check sheet |
| `streamlit run app.py` | 9 | Opens the dashboard |

To use only the downloaded files so far, run `python src/get_data.py --combine-only`, then continue from `prepare.py`.

## Data

- **Usage:** [Azure Public Dataset V2](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md), 30 days of first-party Azure VMs. Please cite it as its repository asks.
- **Prices:** [Azure Retail Prices API](https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices), Dsv5 Linux on-demand in `eastus`.
