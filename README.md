
---

# ML Infra Studio

A lightweight ML workflow and backtesting platform — training, evaluation, and performance benchmarking across models. Demonstrates core ML infrastructure: job orchestration, distributed pull-based workers with heartbeats, dataset ingestion with schema handling, artifact storage, and quant-style backtesting.

---

## Architecture

```
[ Streamlit Dashboard ]  ← submit, inspect, compare, dataset explorer, worker monitor
         │
         ▼
[ Go Orchestrator ]      ← job queue, lifecycle, worker registry, artifact registry, dataset metadata cache
         │
         ▼
[ Python Worker(s) ]     ← heartbeat loop, dataset loading + encoding, train / evaluate / backtest pipelines
         │
    [ libeval.so ]       ← C++ acceleration: accuracy, F1, rolling stats, Sharpe, drawdown
```

---

## What's in each file

| File              | Lang   | Role                                                                                                            |
| ----------------- | ------ | --------------------------------------------------------------------------------------------------------------- |
| `orchestrator.go` | Go     | Control plane — queue, worker pool, heartbeats, artifact refs, dataset registry, metadata caching               |
| `worker.py`       | Python | Pulls jobs, heartbeats, loads datasets (numeric + categorical), runs sklearn/PyTorch pipelines, saves artifacts |
| `eval.cpp`        | C++    | Shared lib — accuracy, F1, rolling stats, Sharpe ratio, max drawdown                                            |
| `dashboard.py`    | Python | Streamlit UI — runs, inspection, metrics, backtest analysis, dataset explorer, worker monitoring                |

---

## Models supported

| Model           | Backend                                             |
| --------------- | --------------------------------------------------- |
| `random_forest` | sklearn RandomForestClassifier                      |
| `xgboost`       | sklearn GradientBoostingClassifier                  |
| `logistic`      | sklearn LogisticRegression                          |
| `svm`           | sklearn SVC                                         |
| `pytorch_mlp`   | PyTorch MLP with configurable hidden dims + dropout |

---

## Dataset Handling

* Supports **CSV dataset upload via dashboard**
* Automatic **schema detection**:

  * numeric features → used directly
  * categorical features → label-encoded at runtime
* Continuous targets are automatically converted to classification via **median thresholding**
* Dataset metadata (rows, features, class balance) is **computed once at upload and cached**
* Avoids repeated full-file scans for fast UI performance

---

## Job Types

* **train** — fits model, reports:

  * accuracy, F1, precision, recall
  * epoch loss curve
  * confusion matrix
  * feature importance (tree-based models)

* **evaluate** — scores model on held-out data

* **backtest** — evaluates across N rolling windows:

  * per-window accuracy/F1
  * mean + std dev (stability proxy)

---

## Dashboard Tabs

* **Runs** — filterable run table by status/type

* **Inspect Run** — full run detail:

  * config
  * metrics
  * epoch loss curve
  * confusion matrix
  * feature importance
  * artifacts

* **Metrics** — cross-run analysis:

  * accuracy vs runtime
  * distribution plots
  * multi-run comparison

* **Backtest** —:

  * per-window bar charts
  * accuracy vs stability visualization

* **Datasets** —:

  * dataset metadata (rows, features, classes)
  * class distribution
  * feature list
  * best model per dataset

* **Workers** —:

  * live worker pool
  * status (idle / busy / offline)
  * jobs handled
  * last heartbeat

---

## Artifact Storage

Workers save artifacts to:

```
./artifacts/<runId>/
```

Includes:

* `model.pkl` — sklearn model
* `model.pt` — PyTorch model
* `eval_report.json` — metrics + config
* `backtest_report.json` — window-level results

Accessible via:

```
GET /artifacts/<runId>/<filename>
```

---

## Worker Model

* Workers poll using:

```
GET /next_job?workerId=X
```

* Send heartbeat every 10s:

```
POST /heartbeat
```

* Marked offline after 30s inactivity

Horizontal scaling:

```bash
WORKER_ID=worker-1 python worker.py &
WORKER_ID=worker-2 python worker.py &
```

Pull-based queue automatically distributes jobs.

---

## Performance Characteristics

Typical runtime (local CPU, ~250k rows dataset):

| Model                   | Runtime       |
| ----------------------- | ------------- |
| logistic                | ~2–15 sec     |
| random_forest           | ~10–60 sec    |
| pytorch_mlp (10 epochs) | ~1–3 min      |
| backtest (3 windows)    | ~30 sec–3 min |

---

## Quick Start

```bash
# 1. Build C++ eval lib (optional)
g++ -O2 -shared -fPIC -o libeval.so eval.cpp

# 2. Go orchestrator
go mod tidy
go run orchestrator.go

# 3. Python deps
pip install scikit-learn numpy requests streamlit plotly pandas torch

# 4. Worker
python worker.py

# 5. Dashboard
streamlit run dashboard.py
```

---

## API

| Method | Endpoint                      | Description       |
| ------ | ----------------------------- | ----------------- |
| POST   | `/submit`                     | Submit job        |
| GET    | `/next_job`                   | Worker polling    |
| PUT    | `/runs/:runId`                | Update run        |
| POST   | `/heartbeat`                  | Worker heartbeat  |
| GET    | `/runs`                       | List runs         |
| GET    | `/runs/:runId`                | Get run           |
| GET    | `/runs/:runId/artifacts`      | List artifacts    |
| GET    | `/artifacts/:runId/:filename` | Download artifact |
| GET    | `/workers`                    | List workers      |
| POST   | `/datasets/upload`            | Upload dataset    |
| GET    | `/datasets`                   | Dataset registry  |
| GET    | `/datasets/:name`             | Dataset metadata  |
| GET    | `/stats`                      | System stats      |
| GET    | `/health`                     | Health check      |

---

## PyTorch MLP Config

```json
{
  "type": "train",
  "config": {
    "model": "pytorch_mlp",
    "dataset": "salary_prediction",
    "epochs": 10,
    "batchSize": 32,
    "learningRate": 0.001,
    "hiddenDims": [64, 32],
    "dropout": 0.3
  }
}
```

---

## Future

* parallel backtest execution across workers
* dataset caching to `.npy` for faster reloads
* regression support (MAE, RMSE, R²)
* experiment tracking (run comparisons + tagging)
* database-backed run store (Postgres / SQLite)
* retry + failure handling
* worker autoscaling
* model serving / inference endpoint

---
