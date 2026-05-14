"""
plots.py — visualizaciones matplotlib compartidas entre pipelines
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report as _skl_report,
                             roc_curve, roc_auc_score)

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Exploración de datos ──────────────────────────────────────────────────────

def plot_class_distribution(vc, label_col: str = "label", save_path: str = None) -> None:
    """Barplot y pie chart de distribución de clases a partir de un value_counts."""
    import seaborn as sns
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.barplot(x=vc.index, y=vc.values, ax=axes[0], palette="Set2")
    axes[0].set_title("Distribución de clases (conteo)")
    axes[0].set_xlabel(label_col.capitalize())
    axes[0].set_ylabel("Nº muestras")
    for bar, val in zip(axes[0].patches, vc.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(vc.values) * 0.01, f"{val:,}",
                     ha="center", va="bottom", fontsize=10)

    axes[1].pie(vc.values, labels=vc.index,
                autopct="%1.1f%%", colors=sns.color_palette("Set2"))
    axes[1].set_title("Proporción de clases")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[plot] guardado → {save_path}")
    plt.show()


def plot_text_length(df, text_col: str = "text", label_col: str = "label",
                     save_path: str = None) -> None:
    """Histogramas de caracteres y palabras por muestra, desglosados por clase."""
    import pandas as _pd
    df = df.copy()
    df["n_chars"] = df[text_col].str.len()
    df["n_words"] = df[text_col].str.split().str.len()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in zip(axes,
                               ["n_chars", "n_words"],
                               ["Caracteres por tweet", "Palabras por tweet"]):
        for label_val in df[label_col].unique():
            ax.hist(df[df[label_col] == label_val][col], bins=40, alpha=0.5, label=label_val)
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[plot] guardado → {save_path}")
    plt.show()


def plot_coherence(coherences: dict, best_k: int) -> None:
    """Gráfica de coherencia c_v en función del número de tópicos K."""
    plt.figure(figsize=(7, 4))
    plt.plot(list(coherences.keys()), list(coherences.values()),
             marker="o", color="#0EA5E9", linewidth=2)
    plt.axvline(best_k, color="#F59E0B", linestyle="--", label=f"mejor K={best_k}")
    plt.xlabel("Número de tópicos (K)")
    plt.ylabel("Coherencia c_v")
    plt.title("Coherencia LDA por número de tópicos")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── Módulo 5 · Sesión 1 ───────────────────────────────────────────────────────

def plot_linear_regression(x_np, y_np, ypred_np, w, b, losses):
    """Ajuste lineal y curva de pérdida MSE (fig0_regresion.png)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.scatter(x_np, y_np, alpha=0.5, color='#5ba3d9', s=25)
    ax1.plot(x_np, ypred_np, color='#e8a020', lw=2.5,
             label=f'ŷ = {w:.2f}·x + {b:.2f}')
    ax1.set_title('Regresión Lineal — ajuste', fontweight='bold')
    ax1.legend(); ax1.spines[['top', 'right']].set_visible(False)
    ax2.plot(losses, color='#2d5f8a', lw=2)
    ax2.set_title('Curva de pérdida (MSE)', fontweight='bold')
    ax2.set_xlabel('Época'); ax2.spines[['top', 'right']].set_visible(False)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'fig0_regresion.png', dpi=110); plt.show()


def plot_mlp_boundary(xx, yy, zz, X_np, y_np, losses):
    """Frontera de decisión del MLP y curva BCE (fig1_mlp.png)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.contourf(xx, yy, zz, alpha=0.35, cmap='coolwarm')
    ax1.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap='coolwarm',
                edgecolors='k', s=25, lw=0.5)
    ax1.set_title('MLP — Frontera de decisión (moons)', fontweight='bold')
    ax1.spines[['top', 'right']].set_visible(False)
    ax2.plot(losses, color='#2d5f8a', lw=2)
    ax2.set_title('BCE Loss durante entrenamiento', fontweight='bold')
    ax2.set_xlabel('Época'); ax2.spines[['top', 'right']].set_visible(False)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'fig1_mlp.png', dpi=110); plt.show()


def plot_rnn_lstm(losses_rnn, losses_lstm):
    """Pérdida RNN vs LSTM y gradiente evanescente (fig2_rnn_lstm.png)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(losses_rnn,  lw=2, color='#c0392b', label='RNN')
    ax1.plot(losses_lstm, lw=2, color='#2d5f8a', label='LSTM', linestyle='--')
    ax1.set_title('RNN vs LSTM — pérdida de entrenamiento', fontweight='bold')
    ax1.set_xlabel('Época'); ax1.legend(); ax1.spines[['top', 'right']].set_visible(False)
    T_g = np.arange(1, 22)
    ax2.plot(T_g, 0.87**T_g, 'o-', color='#c0392b', lw=2, ms=5, label='RNN (desvanece)')
    ax2.plot(T_g, np.clip(0.94**T_g + np.random.randn(len(T_g)) * 0.01, 0.05, 1),
             's--', color='#2d5f8a', lw=2, ms=5, label='LSTM (estable)')
    ax2.set_title('Gradiente evanescente', fontweight='bold')
    ax2.set_xlabel('Pasos BPTT'); ax2.legend(); ax2.spines[['top', 'right']].set_visible(False)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'fig2_rnn_lstm.png', dpi=110); plt.show()


def plot_attention_weights(w_bid_np, w_caus_np, tokens):
    """Heatmaps de atención bidireccional y causal (fig3_atencion.png).

    w_bid_np, w_caus_np: numpy arrays de shape (seq, seq).
    """
    T = len(tokens)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, w, title, cmap in [
        (ax1, w_bid_np,  'Self-attention bidireccional (BERT)', 'Blues'),
        (ax2, w_caus_np, 'Atención causal con máscara (GPT)',   'Oranges'),
    ]:
        im = ax.imshow(w, cmap=cmap, vmin=0)
        ax.set_xticks(range(T)); ax.set_xticklabels(tokens, rotation=30, ha='right')
        ax.set_yticks(range(T)); ax.set_yticklabels(tokens)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Key'); ax.set_ylabel('Query')
        plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'fig3_atencion.png', dpi=110); plt.show()


def plot_model_comparison(losses_rnn, losses_lstm, losses_trf):
    """Comparativa de pérdida RNN · LSTM · Transformer (fig4_comparativa.png)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses_rnn,  lw=2, color='#c0392b', label='RNN',         alpha=0.8)
    ax.plot(losses_lstm, lw=2, color='#e8a020', label='LSTM',        alpha=0.8, linestyle='--')
    ax.plot(losses_trf,  lw=2, color='#2d5f8a', label='Transformer', alpha=0.9, linestyle=':')
    ax.set_title('Comparativa: RNN · LSTM · Transformer', fontweight='bold', fontsize=12)
    ax.set_xlabel('Época'); ax.set_ylabel('Loss')
    ax.legend(fontsize=11); ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'fig4_comparativa.png', dpi=110); plt.show()


# ── Evaluación de clasificadores (clasificación de sentimiento) ───────────────

def plot_report(y_true, y_pred, label_names, history=None, y_score=None,
                title="", save_path=None):
    """
    Muestra en una sola figura:
      - Matriz de confusión normalizada con conteos absolutos
      - Precision / Recall / F1 por clase (barras agrupadas)
      - Curva ROC con AUC              ← solo si se pasa y_score
      - Curvas de loss y accuracy      ← solo si se pasa history
    """
    has_history = bool(history)
    has_roc     = y_score is not None
    n_cols = 3 if has_roc else 2

    if has_roc and has_history:
        height_ratios = [4.5, 4.5]
    elif has_history:
        height_ratios = [4.5, 2.0, 4.5]
    else:
        height_ratios = [4.5, 2.0]
    n_rows = len(height_ratios)

    fig = plt.figure(figsize=(6 * n_cols, sum(height_ratios)), layout="constrained")
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    gs = fig.add_gridspec(n_rows, n_cols, height_ratios=height_ratios)

    _plot_confusion_matrix(y_true, y_pred, label_names, ax=fig.add_subplot(gs[0, 0]))
    _plot_metrics(y_true, y_pred, label_names,          ax=fig.add_subplot(gs[0, 1]))

    if has_roc:
        _plot_roc(y_true, y_score, ax=fig.add_subplot(gs[0, 2]))

    if has_roc and has_history:
        _plot_training_curves(history,
                              ax_loss=fig.add_subplot(gs[1, 0]),
                              ax_acc =fig.add_subplot(gs[1, 1]))
        _plot_text_report(y_true, y_pred, label_names, ax=fig.add_subplot(gs[1, 2]))
    elif has_history:
        _plot_text_report(y_true, y_pred, label_names, ax=fig.add_subplot(gs[1, :]))
        _plot_training_curves(history,
                              ax_loss=fig.add_subplot(gs[2, 0]),
                              ax_acc =fig.add_subplot(gs[2, 1]))
    else:
        _plot_text_report(y_true, y_pred, label_names, ax=fig.add_subplot(gs[1, :]))
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[plot] guardado → {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, label_names,
                          title="Confusion Matrix", save_path=None):
    """Matriz de confusión standalone."""
    fig, ax = plt.subplots(figsize=(5, 4))
    _plot_confusion_matrix(y_true, y_pred, label_names, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_metrics(y_true, y_pred, label_names,
                 title="Classification Metrics", save_path=None):
    """Barras de precision/recall/f1 por clase standalone."""
    fig, ax = plt.subplots(figsize=(6, 4))
    _plot_metrics(y_true, y_pred, label_names, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_comparison(results: dict, title="Comparativa de métodos", save_path=None):
    """Bar chart comparando accuracy entre métodos. results = {name: accuracy}."""
    names, accs = zip(*results.items())
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
              "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, accs, color=colors[:len(names)])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", label="baseline (50%)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_loss_curve(losses, title="Pérdida de entrenamiento", label="train",
                    color='#2d5f8a', save_path=None):
    """Curva de pérdida para un único modelo."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, lw=2, color=color, label=label)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Época'); ax.set_ylabel('Loss')
    ax.legend(); ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def plot_training_curves(history, title="Training Curves", save_path=None):
    """Curvas de loss y accuracy standalone."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title, fontsize=12)
    _plot_training_curves(history, ax_loss=ax_loss, ax_acc=ax_acc)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ── implementaciones internas ─────────────────────────────────────────────────

def _plot_confusion_matrix(y_true, y_pred, label_names, ax):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = len(label_names)
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.60 else "black"
            ax.text(j, i, f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})",
                    ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(n)); ax.set_xticklabels(label_names)
    ax.set_yticks(range(n)); ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")


def _plot_metrics(y_true, y_pred, label_names, ax):
    rep     = _skl_report(y_true, y_pred, target_names=label_names, output_dict=True)
    metrics = ["precision", "recall", "f1-score"]
    x, w    = np.arange(len(label_names)), 0.25

    for i, metric in enumerate(metrics):
        values = [rep[name][metric] for name in label_names]
        ax.bar(x + i * w, values, w, label=metric)

    ax.set_xticks(x + w); ax.set_xticklabels(label_names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(f"Metrics  (accuracy = {rep['accuracy']:.3f})")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)


def _plot_training_curves(history, ax_loss, ax_acc):
    epochs  = [h["epoch"]   for h in history]
    tr_loss = [h["tr_loss"] for h in history]
    vl_loss = [h["vl_loss"] for h in history]
    tr_acc  = [h["tr_acc"]  for h in history]
    vl_acc  = [h["vl_acc"]  for h in history]

    for ax, tr, vl, ylabel, ttl in [
        (ax_loss, tr_loss, vl_loss, "Loss",     "Training Loss"),
        (ax_acc,  tr_acc,  vl_acc,  "Accuracy", "Training Accuracy"),
    ]:
        ax.plot(epochs, tr, label="train", marker="o", markersize=3)
        ax.plot(epochs, vl, label="val",   marker="o", markersize=3)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(ttl)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.xaxis.get_major_locator().set_params(integer=True)


def _plot_roc(y_true, y_score, ax):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    ax.plot(fpr, tpr, lw=1.5, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)


def plot_embedding_tsne(coords, words, colors,
                        title="t-SNE de embeddings", save_name="tsne.png"):
    """Scatter 2D de una proyección t-SNE ya calculada.

    Parameters
    ----------
    coords    : np.ndarray (n_words, 2)  salida de TSNE.fit_transform()
    words     : list[str]  etiquetas de cada punto
    colors    : list[str]  color por palabra (mismo orden)
    title     : título del gráfico
    save_name : nombre del fichero en OUTPUT_DIR
    """
    plt.figure(figsize=(9, 6))
    for (x, y), word, color in zip(coords, words, colors):
        plt.scatter(x, y, color=color, s=60)
        plt.annotate(word, (x, y), fontsize=11,
                     textcoords="offset points", xytext=(6, 4))
    plt.title(title, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    save_path = OUTPUT_DIR / save_name
    plt.savefig(save_path, dpi=150)
    print(f"[plot] guardado → {save_path}")
    plt.show()


def _plot_text_report(y_true, y_pred, label_names, ax):
    report_str = _skl_report(y_true, y_pred, target_names=label_names)
    ax.text(0.5, 0.5, report_str, transform=ax.transAxes,
            ha="center", va="center", fontsize=9,
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.6, pad=0.8))
    ax.set_title("Classification Report", fontsize=10, pad=4)
    ax.axis("off")
