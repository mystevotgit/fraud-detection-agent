from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud_agent.data.load_data import load_creditcard_dataset
from fraud_agent.models.train import stratified_train_valid_test_split


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_ga_convergence(ga_path: Path, out_dir: Path) -> None:
    payload = json.loads(ga_path.read_text(encoding="utf-8"))
    history = pd.DataFrame(payload.get("history", []))
    if history.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history['generation'], history['best_cost'], label='best_cost')
    ax.plot(history['generation'], history['mean_cost'], label='mean_cost')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Cost')
    ax.set_title('GA Convergence (cost)')
    ax.legend()
    out = out_dir / 'ga_convergence.png'
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_feature_importance(model_path: Path, feature_columns: list[str], out_dir: Path) -> None:
    # Only supports sklearn-style feature_importances_
    try:
        import joblib

        model = joblib.load(model_path)
    except Exception:
        return
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # flatten and take absolute
        importances = np.abs(np.ravel(model.coef_))[: len(feature_columns)]
    if importances is None:
        return
    idx = np.argsort(importances)[::-1]
    names = [feature_columns[i] for i in idx]
    vals = importances[idx]
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.15)))
    ax.barh(range(len(names)), vals[::-1])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel('Importance')
    ax.set_title('Feature importances')
    out = out_dir / 'feature_importances.png'
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_eda(raw_path: Path, cfg: dict, out_dir: Path) -> None:
    df = load_creditcard_dataset(str(raw_path))
    # Amount distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df['Amount'].clip(upper=np.percentile(df['Amount'], 99)), bins=50)
    ax.set_title('Amount distribution (99th percentile clipped)')
    ax.set_xlabel('Amount')
    out = out_dir / 'amount_distribution.png'
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    # Class balance
    fig, ax = plt.subplots(figsize=(6, 4))
    df['Class'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Class balance')
    out = out_dir / 'class_balance.png'
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    # Score distribution on test split if possible
    split_cfg = cfg['data']['stratified_split']
    split = stratified_train_valid_test_split(
        df,
        train_size=float(split_cfg['train_size']),
        validation_size=float(split_cfg['validation_size']),
        test_size=float(split_cfg['test_size']),
        random_seed=int(cfg['project']['random_seed']),
    )
    if 'probability' in split.X_test.columns:
        probs = split.X_test['probability']
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(probs, bins=50)
        ax.set_title('Saved probability scores (test split)')
        out = out_dir / 'score_distribution.png'
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)


def main() -> None:
    base = Path('artifacts')
    metrics_dir = base / 'metrics'
    models_dir = base / 'models'
    figures_dir = base / 'figures'
    ensure_dir(figures_dir)

    import yaml

    cfg = yaml.safe_load(Path('configs/default.yaml').read_text(encoding='utf-8')) if Path('configs/default.yaml').exists() else {}

    ga_path = metrics_dir / 'ga_thresholds.json'
    if ga_path.exists():
        plot_ga_convergence(ga_path, figures_dir)

    # Use model referenced by ga payload if present
    model_path = None
    if ga_path.exists():
        payload = json.loads(ga_path.read_text(encoding='utf-8'))
        model_path = models_dir / f"{payload.get('model_name')}.joblib"

    # Attempt feature_importance plot
    if model_path and model_path.exists():
        # derive feature columns from dataset header if available
        try:
            df = load_creditcard_dataset('data/raw/creditcard.csv')
            feature_columns = [c for c in df.columns if c not in ('Class',)]
        except Exception:
            feature_columns = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
        plot_feature_importance(model_path, feature_columns, figures_dir)

    # EDA plots
    raw_path = Path('data/raw/creditcard.csv')
    if raw_path.exists():
        import yaml

        cfg_yaml = yaml.safe_load(Path('configs/default.yaml').read_text(encoding='utf-8')) if Path('configs/default.yaml').exists() else {}
        plot_eda(raw_path, cfg_yaml, figures_dir)


if __name__ == '__main__':
    main()
