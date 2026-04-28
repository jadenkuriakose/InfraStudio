import time
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

baseUrl = "http://localhost:8080"

st.set_page_config(page_title="ML Infra Studio", layout="wide", page_icon="⚙️")

st.markdown("""
<style>
    .stMetric { background: #1a1a2e; border-radius: 8px; padding: 12px; }
    .status-badge { font-size: 12px; font-weight: 600; padding: 2px 8px; border-radius: 4px; }
    div[data-testid="stSidebarContent"] { background: #0f0f1a; }
</style>
""", unsafe_allow_html=True)

st.title("⚙️ ML Infra Studio")

def get(path: str, params: dict = None):
    try:
        r = requests.get(f"{baseUrl}{path}", params=params, timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None

def post(path: str, payload: dict):
    try:
        r = requests.post(f"{baseUrl}{path}", json=payload, timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None

def uploadDataset(uploadedFile, datasetName, labelCol, description):
    files = {"file": (uploadedFile.name, uploadedFile.getvalue(), "text/csv")}
    data = {
        "name": datasetName.strip() or uploadedFile.name.replace(".csv", ""),
        "labelCol": labelCol.strip() or "label",
        "description": description.strip() or "Uploaded CSV dataset.",
    }
    try:
        r = requests.post(f"{baseUrl}/datasets/upload", files=files, data=data, timeout=20)
        if r.ok:
            return True, r.json()
        return False, r.text
    except Exception as e:
        return False, str(e)

if "refreshInterval" not in st.session_state:
    st.session_state.refreshInterval = 5
if "lastRefresh" not in st.session_state:
    st.session_state.lastRefresh = 0
if "selectedRun" not in st.session_state:
    st.session_state.selectedRun = None

with st.sidebar:
    st.header("⚙️ Submit Job")

    with st.expander("📤 Import Dataset", expanded=False):
        uploadedFile = st.file_uploader("CSV file", type=["csv"])
        datasetName = st.text_input("Dataset name", value="")
        labelCol = st.text_input("Label column", value="label")
        description = st.text_area("Description", value="Uploaded CSV dataset.", height=80)
        if uploadedFile is not None:
            try:
                previewDf = pd.read_csv(uploadedFile)
                uploadedFile.seek(0)
                st.caption(f"{previewDf.shape[0]} rows × {previewDf.shape[1]} columns")
                st.dataframe(previewDf.head(5), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Preview failed: {e}")
        if uploadedFile is not None and st.button("Upload Dataset", use_container_width=True):
            ok, result = uploadDataset(uploadedFile, datasetName, labelCol, description)
            if ok:
                st.success(f"Uploaded `{result.get('name', datasetName or uploadedFile.name)}`")
                st.rerun()
            else:
                st.error(result)

    datasets = get("/datasets") or []
    datasetNames = [d["name"] for d in datasets]

    if datasetNames:
        selectedDataset = st.selectbox("Dataset", datasetNames, key="dsSelect")
    else:
        selectedDataset = None
        st.warning("No datasets available. Start the orchestrator or upload a CSV.")

    if selectedDataset:
        dsInfo = next((d for d in datasets if d["name"] == selectedDataset), None)
        if dsInfo:
            with st.expander("📊 Dataset Info", expanded=False):
                st.caption(dsInfo.get("description", ""))
                cols = st.columns(4)
                cols[0].metric("Rows", dsInfo.get("rows", 0))
                cols[1].metric("Features", dsInfo.get("features", 0))
                cols[2].metric("Classes", dsInfo.get("classes", 0))
                cols[3].metric("Source", dsInfo.get("source", "—"))
                balance = dsInfo.get("classBalance", {})
                if balance:
                    fig = px.bar(
                        x=list(balance.keys()),
                        y=list(balance.values()),
                        labels={"x": "Class", "y": "Ratio"},
                        height=150,
                    )
                    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
                    st.plotly_chart(fig, use_container_width=True)
                featureNames = dsInfo.get("featureNames", [])
                if featureNames:
                    st.caption("Features: " + ", ".join(featureNames))

    jobType = st.selectbox("Job Type", ["train", "evaluate", "backtest"])
    modelOptions = ["random_forest", "xgboost", "logistic", "svm", "pytorch_mlp"]
    model = st.selectbox("Model", modelOptions)

    with st.expander("🔧 Hyperparameters", expanded=True):
        epochs = st.slider("Epochs", 1, 50, 10)
        batchSize = st.select_slider("Batch Size", [16, 32, 64, 128], value=32)
        if model == "pytorch_mlp":
            learningRate = st.select_slider("Learning Rate", [0.0001, 0.001, 0.01, 0.1], value=0.001)
            hiddenLayer1 = st.slider("Hidden Layer 1", 16, 256, 64, step=16)
            hiddenLayer2 = st.slider("Hidden Layer 2", 0, 128, 32, step=16)
            dropout = st.slider("Dropout", 0.0, 0.6, 0.3, step=0.1)
            hiddenDims = [hiddenLayer1] + ([hiddenLayer2] if hiddenLayer2 > 0 else [])
        else:
            learningRate = 0.001
            hiddenDims = []
            dropout = 0.0

        if jobType == "backtest":
            windows = st.slider("Windows", 2, 10, 3)
        else:
            windows = 3

    submitDisabled = selectedDataset is None

    if st.button("▶ Submit Job", use_container_width=True, type="primary", disabled=submitDisabled):
        payload = {
            "type": jobType,
            "config": {
                "model": model,
                "dataset": selectedDataset,
                "epochs": epochs,
                "batchSize": batchSize,
                "learningRate": learningRate,
                "hiddenDims": hiddenDims,
                "dropout": dropout,
                "params": {"windows": windows},
            },
        }
        result = post("/submit", payload)
        if result:
            st.success(f"Queued `{result['runId'][:8]}…`")
        else:
            st.error("Orchestrator unreachable")

    st.divider()
    st.subheader("🔄 Auto-Refresh")
    autoRefresh = st.toggle("Enable", value=False)
    if autoRefresh:
        st.session_state.refreshInterval = st.slider("Interval (s)", 2, 30, 5)
    if st.button("Refresh Now", use_container_width=True):
        st.rerun()

runs = get("/runs") or []
workers = get("/workers") or []
stats = get("/stats") or {}

statusColors = {"queued": "🟡", "running": "🔵", "completed": "🟢", "failed": "🔴"}

colA, colB, colC, colD, colE = st.columns(5)
byStatus = stats.get("byStatus", {})
colA.metric("Total Runs", stats.get("totalRuns", 0))
colB.metric("Queued", byStatus.get("queued", 0))
colC.metric("Running", byStatus.get("running", 0))
colD.metric("Completed", byStatus.get("completed", 0))
activeWorkers = sum(1 for w in workers if w.get("status") != "offline")
colE.metric("Active Workers", f"{activeWorkers}/{len(workers)}")

st.divider()

tabRuns, tabInspect, tabMetrics, tabBacktest, tabDatasets, tabWorkers = st.tabs([
    "📋 Runs", "🔍 Inspect Run", "📈 Metrics", "📉 Backtest", "🗂️ Datasets", "🖥️ Workers"
])

def runsDataframe(runList: list) -> pd.DataFrame:
    rows = []
    for r in runList:
        m = r.get("metrics", {})
        rows.append({
            "runId": r["runId"][:8],
            "fullId": r["runId"],
            "type": r["type"],
            "model": r["config"].get("model", "—"),
            "dataset": r["config"].get("dataset", "—"),
            "status": f"{statusColors.get(r['status'], '')} {r['status']}",
            "rawStatus": r["status"],
            "accuracy": m.get("accuracy"),
            "f1": m.get("f1"),
            "loss": m.get("loss"),
            "runtimeMs": m.get("runtimeMs"),
            "latencyMs": m.get("latencyMs"),
            "throughput": m.get("throughput"),
            "workerId": r.get("workerId", "—"),
            "createdAt": r["createdAt"],
        })
    return pd.DataFrame(rows)

with tabRuns:
    colFilter1, colFilter2, colFilter3 = st.columns([2, 2, 4])
    filterStatus = colFilter1.selectbox("Filter Status", ["all", "queued", "running", "completed", "failed"])
    filterType = colFilter2.selectbox("Filter Type", ["all", "train", "evaluate", "backtest"])

    filtered = [
        r for r in runs
        if (filterStatus == "all" or r["status"] == filterStatus)
        and (filterType == "all" or r["type"] == filterType)
    ]

    if not filtered:
        st.info("No runs yet. Submit a job from the sidebar.")
    else:
        df = runsDataframe(filtered)
        visibleCols = [
            "runId", "type", "model", "dataset", "status", "accuracy", "f1",
            "loss", "runtimeMs", "latencyMs", "throughput", "workerId"
        ]
        st.dataframe(
            df[visibleCols],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"{len(filtered)} runs shown")

        runIds = [r["runId"] for r in filtered]
        selected = st.selectbox("Select run to inspect", ["—"] + [r[:8] + "…" for r in runIds])
        if selected != "—":
            idx = [r[:8] + "…" for r in runIds].index(selected)
            st.session_state.selectedRun = runIds[idx]
            st.info(f"Inspecting run `{st.session_state.selectedRun[:8]}` — see **Inspect Run** tab")

with tabInspect:
    if not st.session_state.selectedRun:
        st.info("Select a run from the Runs tab to inspect it.")
    else:
        runId = st.session_state.selectedRun
        run = get(f"/runs/{runId}")
        if not run:
            st.error("Run not found")
        else:
            st.subheader(f"Run `{runId[:8]}`")
            colL, colR = st.columns([1, 2])

            with colL:
                st.markdown("**Config**")
                cfg = run["config"]
                st.json(cfg)
                st.markdown("**Metadata**")
                st.write(f"Type: `{run['type']}`")
                st.write(f"Status: {statusColors.get(run['status'], '')} `{run['status']}`")
                st.write(f"Worker: `{run.get('workerId', '—')}`")
                if run.get("startedAt") and run.get("finishedAt"):
                    from datetime import datetime
                    start = datetime.fromisoformat(run["startedAt"].replace("Z", "+00:00"))
                    end = datetime.fromisoformat(run["finishedAt"].replace("Z", "+00:00"))
                    st.write(f"Duration: `{(end - start).total_seconds():.1f}s`")
                if run.get("error"):
                    st.error(run["error"])

            with colR:
                m = run.get("metrics", {})
                if m:
                    st.markdown("**Metrics**")
                    mCols = st.columns(4)
                    mCols[0].metric("Accuracy", f"{m.get('accuracy', 0):.4f}")
                    mCols[1].metric("F1", f"{m.get('f1', 0):.4f}")
                    mCols[2].metric("Precision", f"{m.get('precision', 0):.4f}")
                    mCols[3].metric("Recall", f"{m.get('recall', 0):.4f}")

                    sCols = st.columns(3)
                    sCols[0].metric("Runtime (ms)", m.get("runtimeMs", "—"))
                    sCols[1].metric("Latency (µs)", m.get("latencyMs", "—"))
                    sCols[2].metric("Throughput", f"{m.get('throughput', 0):.1f}/s")

                    epochCurve = m.get("epochCurve", [])
                    if epochCurve:
                        fig = px.line(
                            x=list(range(1, len(epochCurve) + 1)),
                            y=epochCurve,
                            labels={"x": "Epoch", "y": "Loss"},
                            title="Training Loss Curve",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    confMatrix = m.get("confMatrix", [])
                    if confMatrix:
                        labels = ["Negative", "Positive"]
                        fig = px.imshow(
                            confMatrix,
                            text_auto=True,
                            color_continuous_scale="Blues",
                            x=labels,
                            y=labels,
                            title="Confusion Matrix",
                            labels={"x": "Predicted", "y": "Actual"},
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    featureImp = m.get("featureImportance", {})
                    if featureImp:
                        sortedFeat = sorted(featureImp.items(), key=lambda x: x[1], reverse=True)
                        fig = px.bar(
                            x=[v for _, v in sortedFeat],
                            y=[k for k, _ in sortedFeat],
                            orientation="h",
                            title="Feature Importance",
                            labels={"x": "Importance", "y": "Feature"},
                        )
                        st.plotly_chart(fig, use_container_width=True)

            artifacts = run.get("artifacts", [])
            if artifacts:
                st.markdown("**Artifacts**")
                for art in artifacts:
                    filename = os.path.basename(art["path"]) if art.get("path") else "—"
                    st.write(f"📁 `{filename}` — {art.get('type')} — {art.get('sizeBytes', 0):,} bytes")

with tabMetrics:
    completed = [r for r in runs if r["status"] == "completed"]
    if not completed:
        st.info("No completed runs yet.")
    else:
        df = runsDataframe(completed)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                df,
                x="runId",
                y="accuracy",
                color="model",
                title="Accuracy by Run",
                labels={"runId": "Run", "accuracy": "Accuracy"},
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(
                df,
                x="runtimeMs",
                y="accuracy",
                color="type",
                size="throughput",
                hover_data=["model", "dataset", "runId"],
                title="Accuracy vs Runtime",
            )
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = go.Figure()
            for metric in ["accuracy", "f1"]:
                if metric in df:
                    fig.add_trace(go.Box(y=df[metric].dropna(), name=metric))
            fig.update_layout(title="Model Quality Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            fig = go.Figure()
            for metric in ["runtimeMs", "latencyMs"]:
                if metric in df:
                    fig.add_trace(go.Box(y=df[metric].dropna(), name=metric))
            fig.update_layout(title="System Metrics Distribution")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Compare Runs")
        compareIds = st.multiselect("Select runs", df["runId"].tolist())
        if len(compareIds) >= 2:
            compareRuns = [r for r in completed if r["runId"][:8] in compareIds]
            compareRows = []
            for r in compareRuns:
                m = r.get("metrics", {})
                compareRows.append({
                    "runId": r["runId"][:8],
                    "model": r["config"].get("model"),
                    "accuracy": m.get("accuracy"),
                    "f1": m.get("f1"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "runtimeMs": m.get("runtimeMs"),
                })
            st.dataframe(pd.DataFrame(compareRows), use_container_width=True, hide_index=True)

with tabBacktest:
    backtestRuns = [r for r in runs if r["type"] == "backtest" and r["status"] == "completed"]
    if not backtestRuns:
        st.info("Submit a backtest job to see analysis here.")
    else:
        st.subheader("Backtest Results")
        for run in backtestRuns[:3]:
            m = run.get("metrics", {})
            windowStats = m.get("windowStats", [])
            with st.expander(f"Run `{run['runId'][:8]}` — {run['config'].get('model')} on {run['config'].get('dataset')}"):
                mCols = st.columns(4)
                mCols[0].metric("Mean Accuracy", f"{m.get('accuracy', 0):.4f}")
                mCols[1].metric("Acc Std Dev", f"{m.get('loss', 0):.4f}")
                mCols[2].metric("Mean F1", f"{m.get('f1', 0):.4f}")
                mCols[3].metric("Total Runtime (ms)", m.get("runtimeMs", "—"))

                if windowStats:
                    wdf = pd.DataFrame(windowStats)
                    fig = make_subplots(
                        rows=1,
                        cols=2,
                        subplot_titles=("Accuracy per Window", "F1 per Window"),
                    )
                    fig.add_trace(go.Bar(x=wdf["window"], y=wdf["accuracy"], name="accuracy"), row=1, col=1)
                    fig.add_trace(go.Bar(x=wdf["window"], y=wdf["f1"], name="f1"), row=1, col=2)
                    fig.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(wdf, use_container_width=True, hide_index=True)

        if len(backtestRuns) > 1:
            st.subheader("Cross-Run Comparison")
            rows = []
            for r in backtestRuns:
                m = r.get("metrics", {})
                rows.append({
                    "runId": r["runId"][:8],
                    "model": r["config"].get("model"),
                    "dataset": r["config"].get("dataset"),
                    "meanAccuracy": m.get("accuracy"),
                    "accStd": m.get("loss"),
                    "meanF1": m.get("f1"),
                    "runtimeMs": m.get("runtimeMs"),
                })
            fig = px.scatter(
                pd.DataFrame(rows),
                x="meanAccuracy",
                y="accStd",
                color="model",
                size="runtimeMs",
                hover_data=["runId", "dataset"],
                title="Accuracy vs Stability (lower std = more stable)",
            )
            st.plotly_chart(fig, use_container_width=True)

with tabDatasets:
    st.subheader("Dataset Registry")

    with st.expander("📤 Import New CSV Dataset", expanded=False):
        uploadedFileTab = st.file_uploader("CSV file", type=["csv"], key="datasetTabUploader")
        datasetNameTab = st.text_input("Dataset name", value="", key="datasetTabName")
        labelColTab = st.text_input("Label column", value="label", key="datasetTabLabel")
        descriptionTab = st.text_area("Description", value="Uploaded CSV dataset.", height=80, key="datasetTabDesc")
        if uploadedFileTab is not None:
            try:
                previewDfTab = pd.read_csv(uploadedFileTab)
                uploadedFileTab.seek(0)
                st.caption(f"{previewDfTab.shape[0]} rows × {previewDfTab.shape[1]} columns")
                st.dataframe(previewDfTab.head(10), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Preview failed: {e}")
        if uploadedFileTab is not None and st.button("Upload Dataset", use_container_width=True, key="datasetTabUploadBtn"):
            ok, result = uploadDataset(uploadedFileTab, datasetNameTab, labelColTab, descriptionTab)
            if ok:
                st.success(f"Uploaded `{result.get('name', datasetNameTab or uploadedFileTab.name)}`")
                st.rerun()
            else:
                st.error(result)

    datasets = get("/datasets") or []

    if not datasets:
        st.error("Could not reach orchestrator or no datasets are registered.")
    else:
        for d in datasets:
            sourceIcon = "⬆️" if d.get("source") == "uploaded" else "🧪"
            with st.expander(f"{sourceIcon} {d['name']}", expanded=False):
                st.caption(d.get("description", ""))
                cols = st.columns(5)
                cols[0].metric("Rows", d.get("rows", 0))
                cols[1].metric("Features", d.get("features", 0))
                cols[2].metric("Classes", d.get("classes", 0))
                cols[3].metric("Label", d.get("labelCol", "label"))
                cols[4].metric("Source", d.get("source", "—"))

                colL, colR = st.columns(2)
                balance = d.get("classBalance", {})
                with colL:
                    if balance:
                        fig = px.pie(
                            values=list(balance.values()),
                            names=list(balance.keys()),
                            title="Class Distribution",
                            height=250,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                with colR:
                    featureNames = d.get("featureNames", [])
                    if featureNames:
                        st.markdown("**Features**")
                        for i, name in enumerate(featureNames):
                            st.write(f"`{i}` {name}")

                runsOnDataset = [
                    r for r in runs
                    if r["config"].get("dataset") == d["name"] and r["status"] == "completed"
                ]
                if runsOnDataset:
                    st.markdown(f"**{len(runsOnDataset)} completed run(s) on this dataset**")
                    bestAcc = max((r.get("metrics", {}).get("accuracy", 0) for r in runsOnDataset), default=0)
                    bestModel = next(
                        (
                            r["config"].get("model") for r in runsOnDataset
                            if r.get("metrics", {}).get("accuracy", 0) == bestAcc
                        ),
                        "—",
                    )
                    st.write(f"Best accuracy: `{bestAcc:.4f}` by `{bestModel}`")

with tabWorkers:
    if not workers:
        st.info("No workers have registered yet. Start worker.py to see them here.")
    else:
        workerCols = st.columns(min(len(workers), 3))
        for i, w in enumerate(workers):
            with workerCols[i % 3]:
                statusIcon = {"idle": "🟢", "busy": "🔵", "offline": "⚫"}.get(w["status"], "⚪")
                st.markdown(f"### {statusIcon} `{w['workerId']}`")
                st.write(f"Status: **{w['status']}**")
                st.write(f"Jobs handled: **{w.get('jobsHandled', 0)}**")
                if w.get("currentJob"):
                    st.write(f"Current job: `{w['currentJob'][:8]}`")
                lastSeen = w.get("lastSeen", "")
                if lastSeen:
                    st.caption(f"Last seen: {lastSeen[:19]}")

if autoRefresh:
    time.sleep(st.session_state.refreshInterval)
    st.rerun()