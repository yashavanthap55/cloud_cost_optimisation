"""Step 2: download the VM table, sample VMs, reduce CPU readings to hourly values, fetch prices.

Writes data/vm_table.parquet, data/sample_vms.parquet, data/cpu_hourly.parquet and data/prices.csv.
Safe to re-run: finished work is skipped, so an interrupted download resumes where it stopped.

Each readings file covers about 3.7 hours. By default only the first hour of each file is
streamed, which gives about 6.5 sampled hours per day for a quarter of the download.
Use --full to process every reading (about 150 GB).
"""
import argparse
import ctypes
import http.client
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv
import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parent.parent / "data"
PARTS = DATA / "hourly_parts"
BASE = "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v2"
READINGS = BASE + "/trace_data_vm_cpu_readings_vm_cpu_readings-file-{}-of-195.csv.gz"
DAY = 86400
LAST_DAY = 28              # the plan never uses days 29-30
LAST_FILE = 184            # the first file that starts after day 28 is below this
SAMPLE_SIZE = 20000        # keeping more VMs costs no extra download
REGION = "eastus"
PRODUCT = "Virtual Machines Dsv5 Series"   # Linux, on-demand, general purpose
TRACE_CORES = [2, 4, 8, 24]                # sizes in this trace; ">24" has no known size, so it is dropped
PRICED_CORES = [2, 4, 8, 16, 24]           # 16 is a real Dsv5 size that a 24-vCPU VM can move to


def open_stream(url):
    return urllib.request.urlopen(url, timeout=120)


def get_vm_table():
    out = DATA / "vm_table.parquet"
    if out.exists():
        return pd.read_parquet(out)
    gz = DATA / "vmtable.csv.gz"
    if not gz.exists():
        print("downloading VM table (about 440 MB)")
        tmp = gz.with_suffix(".part")
        with open_stream(BASE + "/trace_data_vmtable_vmtable.csv.gz") as resp, open(tmp, "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
        tmp.rename(gz)
    # The trace's own CPU summary columns cover the whole month, so they would leak the
    # test week into the features. They are deliberately not kept.
    names = ["vm_id", "subscription", "deployment", "created", "deleted",
             "max_cpu", "avg_cpu", "p95_cpu", "category", "cores", "memory"]
    table = csv.read_csv(
        gz,
        read_options=csv.ReadOptions(column_names=names),
        convert_options=csv.ConvertOptions(
            include_columns=["vm_id", "created", "deleted", "category", "cores", "memory"],
            column_types={"cores": pa.string(), "memory": pa.string()},
        ),
    )
    pq.write_table(table, out)
    gz.unlink()
    return table.to_pandas()


def sample_vms(vms):
    out = DATA / "sample_vms.parquet"
    if out.exists():
        return pd.read_parquet(out)
    eligible = vms[
        (vms.created == 0)
        & (vms.deleted >= LAST_DAY * DAY)
        & vms.cores.isin([str(c) for c in TRACE_CORES])
    ]
    print(f"{len(eligible):,} VMs ran from the start through day {LAST_DAY}")
    sample = eligible.sample(min(SAMPLE_SIZE, len(eligible)), random_state=42)
    sample.to_parquet(out)
    return sample


def read_one_file(number, vm_ids, full):
    """Stream one readings file, keep the sampled VMs, and save their hourly maximum CPU."""
    out = PARTS / f"{number:03d}.parquet"
    if out.exists():
        return
    options = dict(
        read_options=csv.ReadOptions(
            column_names=["ts", "vm_id", "min_cpu", "max_cpu", "avg_cpu"], block_size=4 << 20
        ),
        convert_options=csv.ConvertOptions(
            include_columns=["ts", "vm_id", "max_cpu"],
            column_types={"ts": pa.int64(), "vm_id": pa.string(), "max_cpu": pa.float32()},
        ),
    )
    for attempt in range(5):
        try:
            batches, stop = [], None
            with open_stream(READINGS.format(number)) as resp:
                reader = csv.open_csv(pa.input_stream(resp, compression="gzip"), **options)
                for batch in reader:
                    if stop is None:
                        window_hour = batch["ts"][0].as_py() // 3600
                        stop = 2**62 if full else batch["ts"][0].as_py() + 3600
                    keep = pc.and_(
                        pc.is_in(batch["vm_id"], value_set=vm_ids),
                        pc.less(batch["ts"], pa.scalar(stop, pa.int64())),
                    )
                    batches.append(batch.filter(keep))
                    if pc.max(batch["ts"]).as_py() >= stop:
                        break
            break
        except (OSError, http.client.HTTPException, pa.ArrowException) as e:
            print(f"file {number}: {e}; retry {attempt + 1}", flush=True)
            time.sleep(30 * (attempt + 1))
    else:
        raise RuntimeError(f"file {number} failed after 5 attempts")

    df = pa.Table.from_batches(batches).to_pandas()
    # A sampled window can straddle two clock hours, so it is labelled by the hour it starts in.
    df["hour"] = df.ts // 3600 if full else window_hour
    hourly = df.groupby(["vm_id", "hour"], as_index=False).max_cpu.max()
    tmp = out.with_suffix(".part")
    hourly.to_parquet(tmp, index=False)
    tmp.rename(out)
    done = len(list(PARTS.glob("*.parquet")))
    print(f"[{time.strftime('%H:%M:%S')}] file {number} done, {done}/{LAST_FILE} files, {len(hourly):,} rows", flush=True)


def combine_parts():
    """Build cpu_hourly.parquet from whichever files have finished so far."""
    cpu = pd.concat(pd.read_parquet(p) for p in sorted(PARTS.glob("*.parquet")))
    cpu = cpu[cpu.hour < LAST_DAY * 24]
    cpu = cpu.groupby(["vm_id", "hour"], as_index=False).max_cpu.max()
    cpu.to_parquet(DATA / "cpu_hourly.parquet", index=False)
    print(f"cpu_hourly: {len(cpu):,} rows, {cpu.hour.nunique()} distinct hours")


def get_cpu(sample, full, workers):
    PARTS.mkdir(exist_ok=True)
    vm_ids = pa.array(sample.vm_id.tolist(), type=pa.string())
    # Every 4th file first, then fill in, so the whole month is covered early and any
    # partial download can already be combined and used.
    order = sorted(range(1, LAST_FILE + 1), key=lambda n: (n % 4, n))
    with ThreadPoolExecutor(workers) as pool:
        for job in [pool.submit(read_one_file, n, vm_ids, full) for n in order]:
            job.result()
    combine_parts()


def get_prices():
    """Price each core bucket at the family's per-vCPU rate.

    Dsv5 has no 24 vCPU size, so the closest-size rule in the plan would tie between 16 and 32.
    The family's price is linear in vCPUs, which is checked here, so the rate prices it cleanly.
    """
    out = DATA / "prices.csv"
    if out.exists():
        return
    query = f"serviceName eq 'Virtual Machines' and armRegionName eq '{REGION}' " \
            f"and priceType eq 'Consumption' and productName eq '{PRODUCT}'"
    url = "https://prices.azure.com/api/retail/prices?" + urllib.parse.urlencode({"$filter": query})
    items = []
    while url:
        with open_stream(url) as resp:
            page = json.load(resp)
        items += page["Items"]
        url = page.get("NextPageLink")

    sizes = {}
    for item in items:
        match = re.fullmatch(r"Standard_D(\d+)s_v5", item["armSkuName"])
        if match and item["unitOfMeasure"] == "1 Hour" \
                and not re.search("Spot|Low Priority", item["meterName"]):
            sizes[int(match.group(1))] = item["retailPrice"]
    if len(sizes) < 3:
        raise RuntimeError(f"expected Dsv5 prices, got {sizes}")

    rates = {cores: price / cores for cores, price in sizes.items()}
    rate = sorted(rates.values())[len(rates) // 2]
    if max(rates.values()) / min(rates.values()) > 1.01:
        raise RuntimeError(f"Dsv5 price is not linear in vCPUs: {rates}")
    prices = pd.DataFrame({"cores": PRICED_CORES})
    prices["hourly_price"] = prices.cores * rate
    prices["family"] = PRODUCT
    prices["region"] = REGION
    prices.to_csv(out, index=False)
    print(prices.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="read every reading, not one hour per file")
    parser.add_argument("--workers", type=int, default=16, help="parallel downloads")
    parser.add_argument("--combine-only", action="store_true", help="use the files finished so far")
    parser.add_argument("--no-readings", action="store_true", help="stop before the CPU readings")
    args = parser.parse_args()

    if sys.platform == "win32":     # keep the PC awake during the download, only while this runs
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    DATA.mkdir(exist_ok=True)
    if args.combine_only:
        combine_parts()
        return
    get_prices()
    vms = get_vm_table()
    print(f"VM table: {len(vms):,} VMs")
    sample = sample_vms(vms)
    print(f"sample: {len(sample):,} VMs")
    if not args.no_readings:
        get_cpu(sample, args.full, args.workers)


if __name__ == "__main__":
    main()
