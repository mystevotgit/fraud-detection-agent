from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from dash import Dash, dcc, html
from dash.dash_table import DataTable
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from fraud_agent.utils.config import load_config
from fraud_agent.data.load_data import load_creditcard_dataset
from fraud_agent.data.validate_data import require_valid_creditcard_dataset
from fraud_agent.models.train import stratified_train_valid_test_split
from fraud_agent.models.evaluate import timed_predict_proba
from fraud_agent.agent.decision_policy import Thresholds, decide_actions
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix


def make_figures(config):
    df = load_creditcard_dataset(config["paths"]["raw_data"])
    require_valid_creditcard_dataset(df)
    split_cfg = config["data"]["stratified_split"]
    split = stratified_train_valid_test_split(
        df,
        train_size=float(split_cfg["train_size"]),
        validation_size=float(split_cfg["validation_size"]),
        test_size=float(split_cfg["test_size"]),
        random_seed=int(config["project"]["random_seed"]),
    )

    metrics_dir = Path(config["paths"]["metrics_dir"])
    thresholds_path = metrics_dir / "ga_thresholds.json"
    payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
    model_path = Path(config["paths"]["models_dir"]) / f"{payload['model_name']}.joblib"
    model = joblib.load(model_path)

    probs, _ = timed_predict_proba(model, split.X_test)

    # Histogram
    df_scores = pd.DataFrame({"probability": probs, "actual": split.y_test})
    df_scores["label"] = df_scores["actual"].map({0: "Legitimate", 1: "Fraud"})
    fig_hist = px.histogram(df_scores, x="probability", color="label", nbins=50, barmode="overlay", histnorm="probability density", template="plotly_white", labels={"probability":"Predicted fraud probability"})
    fig_hist.update_traces(opacity=0.6)

    # ROC and PR
    fpr, tpr, _ = roc_curve(split.y_test, probs)
    precision, recall, _ = precision_recall_curve(split.y_test, probs)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    fig_rocc = make_subplots(rows=1, cols=2, subplot_titles=(f"ROC (AUC={roc_auc:.3f})", f"PR (AUC={pr_auc:.3f})"))
    fig_rocc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"), row=1, col=1)
    fig_rocc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), showlegend=False), row=1, col=1)
    fig_rocc.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR"), row=1, col=2)
    fig_rocc.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig_rocc.update_yaxes(title_text="True Positive Rate", row=1, col=1)
    fig_rocc.update_xaxes(title_text="Recall", row=1, col=2)
    fig_rocc.update_yaxes(title_text="Precision", row=1, col=2)
    fig_rocc.update_layout(height=450, template="plotly_white")

    # Confusion heatmap
    thresholds = Thresholds(float(payload["review_threshold"]), float(payload["block_threshold"]))
    binary_preds = (probs >= thresholds.block).astype(int)
    tn, fp, fn, tp = confusion_matrix(split.y_test, binary_preds, labels=[0, 1]).ravel()
    cm = [[tn, fp], [fn, tp]]
    fig_cm = go.Figure(data=go.Heatmap(z=cm, x=["Not Blocked", "Blocked"], y=["Legitimate", "Fraud"], colorscale="Blues", hoverinfo="z"))
    fig_cm.update_layout(height=350, template="plotly_white")

    # Sample table (first 50 rows)
    df_samples = split.X_test.reset_index(drop=True).copy()
    df_samples["probability"] = probs
    df_samples["actual"] = split.y_test.reset_index(drop=True)
    sample_table = df_samples.head(50)

    # Persist figures for presentation (requires kaleido)
    figures_dir = Path(config["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Use kaleido helper for compatibility across versions
        import kaleido
        # write_fig_sync accepts (fig, path) for this kaleido version
        kaleido.write_fig_sync(fig_hist, str(figures_dir / "score_distribution.png"))
        kaleido.write_fig_sync(fig_rocc, str(figures_dir / "roc_pr_curves.png"))
        kaleido.write_fig_sync(fig_cm, str(figures_dir / "confusion_matrix.png"))

        # Also save a table view as an image
        fig_table = go.Figure(data=[go.Table(
            header=dict(values=list(sample_table.columns), fill_color='paleturquoise', align='left'),
            cells=dict(values=[sample_table[c].astype(str) for c in sample_table.columns], fill_color='lavender', align='left')
        )])
        fig_table.update_layout(height=600)
        kaleido.write_fig_sync(fig_table, str(figures_dir / "sample_table.png"))

        # Save confusion matrices for a few alternate block thresholds
        alt_thresholds = [0.1, 0.5, 0.9]
        for thr in alt_thresholds:
            binary_preds_alt = (probs >= thr).astype(int)
            tn_a, fp_a, fn_a, tp_a = confusion_matrix(split.y_test, binary_preds_alt, labels=[0, 1]).ravel()
            cm_a = [[tn_a, fp_a], [fn_a, tp_a]]
            fig_cm_alt = go.Figure(data=go.Heatmap(z=cm_a, x=["Not Blocked", "Blocked"], y=["Legitimate", "Fraud"], colorscale="Blues", hoverinfo="z"))
            fig_cm_alt.update_layout(title_text=f"Confusion matrix (block threshold={thr})", height=350)
            kaleido.write_fig_sync(fig_cm_alt, str(figures_dir / f"confusion_matrix_block_{int(thr*100)}.png"))
    except Exception:
        # If image export fails, continue without halting the app
        pass

    return fig_hist, fig_rocc, fig_cm, sample_table


def build_app():
    config = load_config("configs/default.yaml")
    fig_hist, fig_rocc, fig_cm, sample_table = make_figures(config)

    app = Dash(__name__)
    app.layout = html.Div([
        html.H1("Fraud Detection Agent — Dash Diagnostics"),
        html.H2("Score distribution by true label"),
        dcc.Graph(figure=fig_hist),
        html.H2("ROC and PR curves"),
        dcc.Graph(figure=fig_rocc),
        html.H2("Confusion matrix at block threshold"),
        dcc.Graph(figure=fig_cm),
        html.H2("Sample-level predictions (first 50 rows)"),
        DataTable(
            data=sample_table.to_dict("records"),
            page_size=10,
            style_table={"overflowX": "auto"},
        ),
    ])
    return app


if __name__ == "__main__":
    app = build_app()
    app.run(debug=True, port=8050)
