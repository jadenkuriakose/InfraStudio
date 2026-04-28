import ctypes
import json
import os
import pickle
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import requests

ORCHESTRATOR = os.getenv("ORCHESTRATOR_URL", "http://localhost:8080")
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "2.0"))
WORKER_ID = os.getenv("WORKER_ID", f"worker-{uuid.uuid4().hex[:6]}")
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "./artifacts")
DATASET_DIR = os.getenv("DATASET_DIR", "./datasets")

evalLib = None
libPath = os.path.join(os.path.dirname(__file__), "libeval.so")
if os.path.exists(libPath):
    try:
        evalLib = ctypes.CDLL(libPath)
        evalLib.computeMetrics.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ]
        evalLib.computeMetrics.restype = None
        print("loaded libeval.so")
    except Exception as e:
        print(f"libeval.so load failed: {e}, using Python fallback")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    print(f"PyTorch {torch.__version__} available")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed — sklearn-only mode")


@dataclass
class jobMetrics:
    accuracy: float = 0.0
    loss: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    runtimeMs: int = 0
    latencyMs: int = 0
    throughput: float = 0.0
    epochCurve: list[float] = field(default_factory=list)
    windowStats: list[dict] = field(default_factory=list)
    confMatrix: list[list[int]] = field(default_factory=list)
    featureImportance: dict[str, float] = field(default_factory=dict)


def loadDataset(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # check for uploaded CSV first
    csvPath = os.path.join(DATASET_DIR, f"{name}.csv")
    if os.path.exists(csvPath):
        return loadCsvDataset(csvPath)

    # built-in synthetic datasets
    configs = {
        "sample_dataset": {"n": 500, "seed": 42, "features": 10},
        "holdout_set":    {"n": 200, "seed": 99, "features": 10},
        "time_series":    {"n": 600, "seed": 7,  "features": 8},
        "imbalanced_set": {"n": 400, "seed": 13, "features": 10, "imbalance": 0.1},
    }
    cfg = configs.get(name, configs["sample_dataset"])
    rng = np.random.default_rng(cfg["seed"])
    nFeatures = cfg["features"]
    n = cfg["n"]
    X = rng.standard_normal((n, nFeatures))

    if "imbalance" in cfg:
        # fixed: use rng not random() so seed is respected
        y = (rng.random(n) < cfg["imbalance"]).astype(int)
    elif name == "time_series":
        # fixed: label is based on local window sign, not cumsum drift
        # cumsum drifts to one class in test split — use sign of short rolling sum instead
        raw = rng.standard_normal(n)
        windowSize = 5
        rollingSum = np.convolve(raw, np.ones(windowSize) / windowSize, mode="same")
        y = (rollingSum > 0).astype(int)
        # verify rough balance — if not, fall back to threshold on features
        if y.mean() > 0.8 or y.mean() < 0.2:
            y = (X[:, 0] + X[:, 1] > 0).astype(int)
    else:
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

    featureNames = [f"feat_{i}" for i in range(nFeatures)]
    if name == "time_series":
        featureNames = ["lag_1", "lag_2", "lag_3", "rolling_mean", "rolling_std", "momentum", "rsi", "volume"][:nFeatures]

    return X, y, featureNames


def loadCsvDataset(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import csv
    metaPath = path.replace(".csv", ".meta.json")
    labelCol = "label"
    if os.path.exists(metaPath):
        with open(metaPath) as f:
            meta = json.load(f)
            labelCol = meta.get("labelCol", "label")

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV at {path} is empty")

    headers = list(rows[0].keys())
    if labelCol not in headers:
        raise ValueError(f"Label column '{labelCol}' not found in {headers}")

    featureNames = [h for h in headers if h != labelCol]
    X = np.array([[float(r[h]) for h in featureNames] for r in rows], dtype=np.float32)
    y = np.array([int(float(r[labelCol])) for r in rows], dtype=np.int64)
    return X, y, featureNames


def computeMetricsFull(predictions: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, float]:
    predictions = np.asarray(predictions).flatten()
    labels = np.asarray(labels).flatten()

    acc = float(np.mean(predictions == labels))
    tp = float(np.sum((predictions == 1) & (labels == 1)))
    fp = float(np.sum((predictions == 1) & (labels == 0)))
    fn = float(np.sum((predictions == 0) & (labels == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # use C++ for acc+f1 if available, keep python precision/recall
    if evalLib is not None:
        n = len(predictions)
        pred_c  = (ctypes.c_double * n)(*predictions.astype(float))
        label_c = (ctypes.c_double * n)(*labels.astype(float))
        acc_c = ctypes.c_double(0.0)
        f1_c  = ctypes.c_double(0.0)
        evalLib.computeMetrics(pred_c, label_c, ctypes.c_int(n), ctypes.byref(acc_c), ctypes.byref(f1_c))
        acc = acc_c.value
        f1  = f1_c.value

    return acc, f1, precision, recall


def confusionMatrix(predictions: np.ndarray, labels: np.ndarray) -> list[list[int]]:
    predictions = np.asarray(predictions).flatten()
    labels = np.asarray(labels).flatten()
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    return [[tn, fp], [fn, tp]]


def saveArtifact(runID: str, name: str, obj: Any, artifactType: str) -> dict:
    dirPath = os.path.join(ARTIFACT_DIR, runID)
    os.makedirs(dirPath, exist_ok=True)
    path = os.path.join(dirPath, name)
    if name.endswith(".pkl"):
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    elif name.endswith(".json"):
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    elif name.endswith(".pt") and TORCH_AVAILABLE:
        torch.save(obj, path)
    sizeBytes = os.path.getsize(path)
    return {
        "runId": runID, "type": artifactType, "path": path,
        "sizeBytes": sizeBytes, "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def buildMlp(inputDim: int, hiddenDims: list[int], dropout: float = 0.3):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            prev = inputDim
            for h in hiddenDims:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return _Net()


def trainSklearn(config: dict[str, Any]) -> tuple[jobMetrics, Any]:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    modelName = config.get("model", "random_forest")
    nEstimators = config.get("epochs", 5) * 10

    baseModels = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True),
        "xgboost":       GradientBoostingClassifier(n_estimators=nEstimators, random_state=42),
        "logistic":      Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]),
        "svm":           Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True))]),
    }
    model = baseModels.get(modelName, baseModels["random_forest"])

    X, y, featureNames = loadDataset(config.get("dataset", "sample_dataset"))

    # fixed: proper train/test split — never evaluate on training data
    splitIdx = int(len(X) * 0.8)
    xTrain, xTest = X[:splitIdx], X[splitIdx:]
    yTrain, yTest = y[:splitIdx], y[splitIdx:]

    t0 = time.perf_counter()
    model.fit(xTrain, yTrain)
    predictions = model.predict(xTest)
    elapsed = time.perf_counter() - t0

    acc, f1, prec, rec = computeMetricsFull(predictions, yTest)
    loss = float(np.mean((predictions != yTest).astype(float)))  # error rate, not MSE
    cm = confusionMatrix(predictions, yTest)

    # feature importance — unwrap pipeline if needed
    baseClf = model.named_steps.get("clf", model) if hasattr(model, "named_steps") else model
    featureImp = {}
    if hasattr(baseClf, "feature_importances_"):
        featureImp = {featureNames[i]: round(float(v), 4) for i, v in enumerate(baseClf.feature_importances_)}

    # epoch curve from GradientBoosting staged scores, or oob for RF
    epochCurve = []
    if hasattr(baseClf, "train_score_"):
        epochCurve = [round(float(s), 4) for s in baseClf.train_score_]
    elif hasattr(baseClf, "oob_score_"):
        epochCurve = [round(float(baseClf.oob_score_), 4)]

    return jobMetrics(
        accuracy=round(acc, 4), loss=round(loss, 4), f1=round(f1, 4),
        precision=round(prec, 4), recall=round(rec, 4),
        runtimeMs=int(elapsed * 1000),
        latencyMs=int((elapsed / max(len(xTest), 1)) * 1e6),
        throughput=round(len(xTest) / max(elapsed, 1e-9), 1),
        confMatrix=cm, featureImportance=featureImp, epochCurve=epochCurve,
    ), model


def trainPytorch(config: dict[str, Any]) -> tuple[jobMetrics, Any]:
    X, y, _ = loadDataset(config.get("dataset", "sample_dataset"))

    splitIdx = int(len(X) * 0.8)
    xTrain = torch.tensor(X[:splitIdx], dtype=torch.float32)
    yTrain = torch.tensor(y[:splitIdx], dtype=torch.float32)
    xTest  = torch.tensor(X[splitIdx:], dtype=torch.float32)
    yTestNp = y[splitIdx:]

    hiddenDims = config.get("hiddenDims") or [64, 32]
    dropout    = config.get("dropout", 0.3)
    lr         = config.get("learningRate", 0.001)
    epochs     = config.get("epochs", 10)
    batchSize  = config.get("batchSize", 32)

    model     = buildMlp(X.shape[1], hiddenDims, dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    epochCurve = []
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        indices   = torch.randperm(len(xTrain))
        epochLoss = 0.0
        nBatches  = 0
        for i in range(0, len(xTrain), batchSize):
            batch = indices[i:i + batchSize]
            optimizer.zero_grad()
            out  = model(xTrain[batch])
            loss = criterion(out, yTrain[batch])
            loss.backward()
            optimizer.step()
            epochLoss += loss.item()
            nBatches  += 1
        epochCurve.append(round(epochLoss / max(nBatches, 1), 4))

    model.eval()
    with torch.no_grad():
        logits = model(xTest)
        preds  = (torch.sigmoid(logits) > 0.5).long().numpy()
    elapsed = time.perf_counter() - t0

    acc, f1, prec, rec = computeMetricsFull(preds, yTestNp)
    cm = confusionMatrix(preds, yTestNp)

    return jobMetrics(
        accuracy=round(acc, 4), loss=round(epochCurve[-1] if epochCurve else 0.0, 4),
        f1=round(f1, 4), precision=round(prec, 4), recall=round(rec, 4),
        runtimeMs=int(elapsed * 1000),
        latencyMs=int((elapsed / max(len(xTest), 1)) * 1e6),
        throughput=round(len(xTest) / max(elapsed, 1e-9), 1),
        epochCurve=epochCurve, confMatrix=cm,
    ), model


def evaluateModel(config: dict[str, Any]) -> tuple[jobMetrics, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    X, y, featureNames = loadDataset(config.get("dataset", "sample_dataset"))

    # fixed: use a proper held-out split, not train==test
    splitIdx = int(len(X) * 0.8)
    xTrain, xTest = X[:splitIdx], X[splitIdx:]
    yTrain, yTest = y[:splitIdx], y[splitIdx:]

    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest  = scaler.transform(xTest)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(xTrain, yTrain)

    t0 = time.perf_counter()
    predictions = model.predict(xTest)
    elapsed = time.perf_counter() - t0

    acc, f1, prec, rec = computeMetricsFull(predictions, yTest)
    featureImp = {featureNames[i]: round(float(v), 4) for i, v in enumerate(model.feature_importances_)}

    return jobMetrics(
        accuracy=round(acc, 4),
        loss=round(float(np.mean((predictions != yTest).astype(float))), 4),
        f1=round(f1, 4), precision=round(prec, 4), recall=round(rec, 4),
        runtimeMs=int(elapsed * 1000),
        latencyMs=int((elapsed / max(len(xTest), 1)) * 1e6),
        throughput=round(len(xTest) / max(elapsed, 1e-9), 1),
        confMatrix=confusionMatrix(predictions, yTest), featureImportance=featureImp,
    ), model


def runBacktest(config: dict[str, Any]) -> tuple[jobMetrics, None]:
    from sklearn.ensemble import RandomForestClassifier

    windows = int((config.get("params") or {}).get("windows", 3))
    X, y, _ = loadDataset(config.get("dataset", "sample_dataset"))

    windowSize   = len(X) // windows
    windowStats  = []
    totalRuntime = 0.0

    for i in range(windows):
        s, e     = i * windowSize, (i + 1) * windowSize
        xSlice, ySlice = X[s:e], y[s:e]
        split    = int(len(xSlice) * 0.8)

        if split < 5 or (len(xSlice) - split) < 5:
            continue

        model = RandomForestClassifier(n_estimators=50, random_state=i)
        model.fit(xSlice[:split], ySlice[:split])

        t0    = time.perf_counter()
        preds = model.predict(xSlice[split:])
        elapsed = time.perf_counter() - t0
        totalRuntime += elapsed

        acc, f1, _, _ = computeMetricsFull(preds, ySlice[split:])
        loss = float(np.mean((preds != ySlice[split:]).astype(float)))
        windowStats.append({"window": i, "accuracy": round(acc, 4), "f1": round(f1, 4), "loss": round(loss, 4)})

    accs = [w["accuracy"] for w in windowStats]
    f1s  = [w["f1"]       for w in windowStats]

    return jobMetrics(
        accuracy=round(float(np.mean(accs)), 4) if accs else 0.0,
        loss=round(float(np.std(accs)), 4) if accs else 0.0,
        f1=round(float(np.mean(f1s)), 4) if f1s else 0.0,
        runtimeMs=int(totalRuntime * 1000),
        latencyMs=int((totalRuntime / max(windows, 1)) * 1000),
        throughput=round(windows / max(totalRuntime, 1e-9), 2),
        windowStats=windowStats,
    ), None


def processJob(job: dict[str, Any]) -> tuple[str, jobMetrics, list[dict], str]:
    jobT      = job.get("type")
    config    = job.get("config", {})
    runID     = job["runId"]
    artifacts = []

    try:
        if jobT == "train":
            modelName = config.get("model", "random_forest")
            if TORCH_AVAILABLE and modelName in ("pytorch_mlp", "mlp"):
                metrics, modelObj = trainPytorch(config)
                art = saveArtifact(runID, "model.pt", modelObj.state_dict(), "model_weights")
            else:
                metrics, modelObj = trainSklearn(config)
                art = saveArtifact(runID, "model.pkl", modelObj, "model_weights")
            artifacts.append(art)
            report = {
                "runId": runID, "model": modelName, "config": config,
                "metrics": asdict(metrics), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            artifacts.append(saveArtifact(runID, "eval_report.json", report, "eval_report"))

        elif jobT == "evaluate":
            metrics, modelObj = evaluateModel(config)
            artifacts.append(saveArtifact(runID, "model.pkl", modelObj, "model_weights"))

        elif jobT == "backtest":
            metrics, _ = runBacktest(config)
            report = {
                "runId": runID, "config": config,
                "windowStats": metrics.windowStats, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            artifacts.append(saveArtifact(runID, "backtest_report.json", report, "backtest_report"))

        else:
            return "failed", jobMetrics(), [], f"unknown job type: {jobT}"

        return "completed", metrics, artifacts, ""

    except Exception:
        import traceback
        return "failed", jobMetrics(), [], traceback.format_exc()


def heartbeatLoop():
    while True:
        try:
            requests.post(f"{ORCHESTRATOR}/heartbeat", json={"workerId": WORKER_ID}, timeout=3)
        except Exception:
            pass
        time.sleep(10)


def pollAndExecute():
    print(f"worker {WORKER_ID} started — polling {ORCHESTRATOR}")
    threading.Thread(target=heartbeatLoop, daemon=True).start()

    while True:
        try:
            resp = requests.get(f"{ORCHESTRATOR}/next_job", params={"workerId": WORKER_ID}, timeout=5)
            if resp.status_code == 204:
                time.sleep(POLL_INTERVAL)
                continue

            job   = resp.json()
            runId = job["runId"]
            print(f"[{WORKER_ID}] executing {job['type']} {runId[:8]} (model={job['config'].get('model')} dataset={job['config'].get('dataset')})")

            status, metrics, artifacts, errMsg = processJob(job)

            requests.put(
                f"{ORCHESTRATOR}/runs/{runId}",
                json={"status": status, "metrics": asdict(metrics), "artifacts": artifacts, "workerId": WORKER_ID, "error": errMsg},
                timeout=5,
            )
            print(f"[{WORKER_ID}] {runId[:8]} → {status}")

        except requests.exceptions.ConnectionError:
            print(f"orchestrator unreachable, retrying in {POLL_INTERVAL}s")
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            print(f"worker error: {e}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)
    pollAndExecute()