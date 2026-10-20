from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

from fraud_agent.agent.cost_model import CostConfig
from fraud_agent.agent.decision_policy import Thresholds
from fraud_agent.agent.explanations import explain_decision
from fraud_agent.features.preprocessing import FEATURE_COLUMNS
from fraud_agent.utils.config import load_config
from fraud_agent.data.load_data import load_creditcard_dataset
from fraud_agent.data.validate_data import require_valid_creditcard_dataset
from fraud_agent.models.train import stratified_train_valid_test_split
from fraud_agent.models.evaluate import timed_predict_proba
from fraud_agent.agent.decision_policy import decide_actions
from streamlit import cache_data


st.set_page_config(page_title="Fraud Detection Agent", layout="wide")
st.title("Cost-Sensitive Fraud Detection Agent")

config = load_config("configs/default.yaml")
metrics_dir = Path(config["paths"]["metrics_dir"])
models_dir = Path(config["paths"]["models_dir"])
thresholds_path = metrics_dir / "ga_thresholds.json"

if not thresholds_path.exists():
    st.warning(
        "No trained artifacts were found. Run the reproducible workflow after proposal approval "
        "and after placing data/raw/creditcard.csv locally."
    )
    st.stop()

threshold_payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
model_path = models_dir / f"{threshold_payload['model_name']}.joblib"
if not model_path.exists():
    st.error(f"Thresholds reference a missing model artifact: {model_path}")
    st.stop()

model = joblib.load(model_path)
thresholds = Thresholds(float(threshold_payload["review_threshold"]), float(threshold_payload["block_threshold"]))

overview, diagnostics, simulator, costs, limitations = st.tabs(
    ["Overview", "Diagnostics", "Transaction Simulator", "Cost Settings", "Limitations"]
)

with overview:
    st.subheader("Saved Agent")
    st.metric("Model", threshold_payload["model_name"])
    st.metric("Review threshold", f"{thresholds.review:.4f}")
    st.metric("Block threshold", f"{thresholds.block:.4f}")
    test_metrics = metrics_dir / "locked_test_metrics.json"
    if test_metrics.exists():
        st.json(json.loads(test_metrics.read_text(encoding="utf-8")))
    else:
        st.info("Locked test metrics have not been generated yet.")

with diagnostics:
    st.subheader("Model & Dataset Diagnostics (test split)")
    st.caption("These visualizations show model scores and outcomes on the held-out test split.")
    # load dataset and split (cache only the split data; avoid caching model)
    raw_path = config["paths"]["raw_data"]
    if not Path(raw_path).exists():
        st.warning("Raw dataset not found; run the reproducible workflow to generate artifacts.")
    else:
        @cache_data
        def get_test_split(raw_path: str, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
            df_full = load_creditcard_dataset(raw_path)
            require_valid_creditcard_dataset(df_full)
            split_cfg = cfg["data"]["stratified_split"]
            split = stratified_train_valid_test_split(
                df_full,
                train_size=float(split_cfg["train_size"]),
                validation_size=float(split_cfg["validation_size"]),
                test_size=float(split_cfg["test_size"]),
                random_seed=int(cfg["project"]["random_seed"]),
            )
            return split.X_test.reset_index(drop=True), split.y_test.reset_index(drop=True)

        X_test, y_test = get_test_split(raw_path, config)
        try:
            probs, _ = timed_predict_proba(model, X_test)
        except Exception as e:
            st.error(f"Failed to compute model probabilities: {e}")
            probs = np.zeros(len(X_test))

        st.markdown("**Score distribution by true label**")
        df_scores = pd.DataFrame({"probability": probs, "actual": y_test})
        df_scores["label"] = df_scores["actual"].map({0: "Legitimate", 1: "Fraud"})
        fig_hist = px.histogram(df_scores, x="probability", color="label", nbins=50, barmode="overlay", histnorm="probability density", template="plotly_white", labels={"probability":"Predicted fraud probability"})
        fig_hist.update_traces(opacity=0.6)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("**ROC and PR curves**")
        fpr, tpr, _ = roc_curve(y_test, probs)
        precision, recall, _ = precision_recall_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"ROC (AUC={roc_auc:.3f})", f"PR (AUC={pr_auc:.3f})"))
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR"), row=1, col=2)
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Operational thresholds, confusion matrix, and sample table**")
        col1, col2 = st.columns([2, 1])
        with col2:
            review_slider = st.slider("Review threshold", 0.0, 1.0, float(thresholds.review), step=0.001)
            block_slider = st.slider("Block threshold", 0.0, 1.0, float(thresholds.block), step=0.001)
            if review_slider >= block_slider:
                st.error("`review` must be smaller than `block` threshold. Adjust sliders.")
        with col1:
            pred_actions = decide_actions(probs, Thresholds(review_slider, block_slider))
            st.write(pd.Series(pred_actions).value_counts())

        # Confusion matrix at block threshold (binary blocked vs not) - Plotly heatmap
        binary_preds = (probs >= block_slider).astype(int)
        try:
            tn, fp, fn, tp = confusion_matrix(y_test, binary_preds, labels=[0, 1]).ravel()
        except Exception:
            tn = fp = fn = tp = 0
        cm = [[tn, fp], [fn, tp]]
        fig_cm = go.Figure(data=go.Heatmap(z=cm, x=["Not Blocked", "Blocked"], y=["Legitimate", "Fraud"], colorscale="Blues", hoverinfo="z"))
        fig_cm.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig_cm, use_container_width=True)

        # Sample-level table with per-sample explanations
        st.markdown("**Sample-level predictions and explanations (test split)**")
        df_samples = X_test.copy()
        df_samples = df_samples.reset_index(drop=True)
        df_samples["probability"] = probs
        df_samples["actual"] = y_test
        # Vectorized action and risk level computation (avoids per-row Python loops)
        actions = decide_actions(df_samples["probability"].to_numpy(), Thresholds(review_slider, block_slider))
        prob_arr = df_samples["probability"].to_numpy()
        risk_level = np.where(prob_arr < 0.05, "low", np.where(prob_arr < 0.5, "moderate", np.where(prob_arr < 0.9, "high", "very high")))
        # Vectorized threshold reasons
        reason_approve = f"Probability is below the review threshold ({review_slider:.3f})."
        reason_manual = f"Probability is between the review threshold ({review_slider:.3f}) and block threshold ({block_slider:.3f})."
        reason_block = f"Probability is at or above the block threshold ({block_slider:.3f})."
        threshold_reason = np.where(actions == "APPROVE", reason_approve, np.where(actions == "MANUAL REVIEW", reason_manual, reason_block))
        df_samples["action"] = actions
        df_samples["risk_level"] = risk_level
        df_samples["threshold_reason"] = threshold_reason

        # Filtering controls
        action_filter = st.multiselect("Filter actions", options=list(pd.Series(df_samples["action"]).unique()), default=list(pd.Series(df_samples["action"]).unique()))
        risk_filter = st.multiselect("Filter risk levels", options=list(pd.Series(df_samples["risk_level"]).unique()), default=list(pd.Series(df_samples["risk_level"]).unique()))
        max_rows = st.number_input("Max rows to show", min_value=10, max_value=1000, value=100, step=10)
        df_view = df_samples[df_samples["action"].isin(action_filter) & df_samples["risk_level"].isin(risk_filter)].head(int(max_rows))
        # Use st.table (static) to avoid potential interactive renderer issues that can crash on some systems.
        st.table(df_view)
        # Provide CSV download for the filtered view
        csv_bytes = df_view.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", data=csv_bytes, file_name="filtered_samples.csv", mime="text/csv")

with simulator:
    st.subheader("Transaction Risk Simulator")
    st.caption("Feature names are anonymized; V1-V28 are PCA-transformed variables from the dataset.")
    values = {}
    col_a, col_b = st.columns(2)
    with col_a:
        values["Time"] = st.number_input("Time", value=0.0)
        values["Amount"] = st.number_input("Amount", min_value=0.0, value=100.0)
    with col_b:
        for idx in range(1, 29):
            values[f"V{idx}"] = st.number_input(f"V{idx}", value=0.0, format="%.6f")

    row = pd.DataFrame([{column: values[column] for column in FEATURE_COLUMNS}])
    probability = float(model.predict_proba(row)[0, 1])
    explanation = explain_decision(probability, thresholds)
    st.metric("Fraud probability", f"{probability:.4f}")
    st.metric("Agent action", explanation.action)
    st.write(explanation.threshold_reason)
    st.info(explanation.note)

with costs:
    st.subheader("Experimental Cost Assumptions")
    cost_config = CostConfig(**config["costs"])
    st.json(cost_config.__dict__)
    st.caption("These are academic assumptions, not real bank cost estimates.")

with limitations:
    st.subheader("Ethics and Limitations")
    st.write(
        "This is an academic prototype. It should not autonomously make real financial decisions. "
        "False positives can inconvenience customers, false negatives can cause financial loss, "
        "historical labels may be incomplete or biased, and PCA-anonymized features limit explanation quality."
    )

