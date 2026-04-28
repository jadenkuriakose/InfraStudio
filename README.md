# ML Infra Studio

A lightweight ML workflow and backtesting platform — training, evaluation, and performance benchmarking across models. Demonstrates core ML infrastructure: job orchestration, distributed pull-based workers with heartbeats, artifact storage, and quant-style backtesting.

---

## Architecture

```
[ Streamlit Dashboard ]  ← submit, inspect, compare, dataset explorer, worker monitor
         │
         ▼
[ Go Orchestrator ]      ← job queue, lifecycle, worker registry, artifact registry, dataset metadata
         │
         ▼
[ Python Worker(s) ]     ← heartbeat loop, train / evaluate / backtest pipelines, artifact saving
         │
    [ libeval.so ]       ← C++ acceleration: accuracy, F1, rolling stats, Sharpe, drawdown
```

---

## What's in each file

| File | Lang | Role |
|---|---|---|
| `orchestrator.go` | Go | Control plane — queue, worker pool, heartbeats, artifact refs, dataset registry |
| `worker.py` | Python | Pulls jobs, heartbeats, runs sklearn/PyTorch pipelines, saves artifacts |
| `eval.cpp` | C++ | Shared lib — accuracy, F1, rolling stats, Sharpe ratio, max drawdown |
| `dashboard.py` | Python | Streamlit UI — 6 tabs: runs, run inspector, metrics, backtest, datasets, workers |

---

## Models supported

| Model | Backend |
|---|---|
| `random_forest` | sklearn RandomForestClassifier |
| `xgboost` | sklearn GradientBoostingClassifier |
| `logistic` | sklearn LogisticRegression |
| `svm` | sklearn SVC |
| `pytorch_mlp` | PyTorch MLP with configurable hidden dims + dropout |

---

## Job Types

- **train** — fits model, reports accuracy/F1/precision/recall + epoch loss curve + confusion matrix + feature importance
- **evaluate** — scores model on held-out data with full metrics
- **backtest** — evaluates across N rolling windows; reports per-window stats + mean/std

---

## Dashboard Tabs

- **Runs** — filterable run table by status and type
- **Inspect Run** — full run detail: config, all metrics, epoch loss curve, confusion matrix, feature importance, artifacts
- **Metrics** — cross-run accuracy/runtime charts, distribution plots, multi-run comparison
- **Backtest** — per-window bar charts, accuracy vs stability scatter across runs
- **Datasets** — dataset explorer with class distribution, feature list, best model per dataset
- **Workers** — live worker pool: status, jobs handled, current job, last heartbeat

---

## Artifact Storage

Workers save artifacts to `./artifacts/<runId>/`:
- `model.pkl` — sklearn model weights
- `model.pt` — PyTorch state dict
- `eval_report.json` — full metrics + config
- `backtest_report.json` — per-window results

Accessible via `GET /artifacts/<runId>/<filename>`.

---

## Worker Heartbeats

Workers send a POST `/heartbeat` every 10 seconds. The orchestrator marks workers **offline** after 30 seconds of silence. The dashboard Workers tab reflects live status.

Run multiple workers in parallel — the pull-based queue distributes jobs automatically:
```bash
WORKER_ID=worker-1 python worker.py &
WORKER_ID=worker-2 python worker.py &
```

---

## Quick Start

```bash
# 1. Build C++ eval lib (optional)
g++ -O2 -shared -fPIC -o libeval.so eval.cpp

# 2. Go deps + orchestrator
go mod tidy
go run orchestrator.go

# 3. Python deps
pip install scikit-learn numpy requests streamlit plotly pandas torch

# 4. Worker (new terminal)
python worker.py

# 5. Dashboard (new terminal)
streamlit run dashboard.py
```

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/submit` | Submit a job |
| `GET` | `/next_job?workerId=X` | Worker polls for next queued job |
| `PUT` | `/runs/:runId` | Worker reports result + artifacts |
| `POST` | `/heartbeat` | Worker heartbeat |
| `GET` | `/runs` | List runs (optional `?status=&type=` filters) |
| `GET` | `/runs/:runId` | Get single run |
| `GET` | `/runs/:runId/artifacts` | List artifacts for a run |
| `GET` | `/artifacts/:runId/:filename` | Download artifact file |
| `GET` | `/workers` | List registered workers |
| `GET` | `/datasets` | List datasets with metadata |
| `GET` | `/datasets/:name` | Get single dataset info |
| `GET` | `/stats` | System stats: queue depth, run counts by status |
| `GET` | `/health` | Health check |

---

## PyTorch MLP Config

```json
{
  "type": "train",
  "config": {
    "model": "pytorch_mlp",
    "dataset": "sample_dataset",
    "epochs": 20,
    "batchSize": 32,
    "learningRate": 0.001,
    "hiddenDims": [64, 32],
    "dropout": 0.3
  }
}
```

---

## Future

- database-backed run store (SQLite / Postgres)
- retry + failure handling with exponential backoff
- rolling window and time-split backtest modes
- worker autoscaling
- model serving endpoint