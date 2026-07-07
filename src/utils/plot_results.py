"""
Generazione delle figure del report a partire dagli storici dei run.

Legge i file ``history.json`` prodotti dal clustering iterativo in
``experiments/logs/<run>/`` e genera in ``figures/`` le immagini PNG
usate da ``docs/REPORT.md`` e dalla presentazione. Ogni figura è
autosufficiente (titolo, assi etichettati, legenda) per poter essere
riusata da sola in una slide.

Uso (dalla radice del repository):
    python -m src.utils.plot_results
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MaxNLocator

# --- Palette (validata, vedi report): serie categoriche in ordine fisso,
# rampa ordinale blu per le scale del dataset (più scuro = più dati) ---
BLUE, AQUA, YELLOW, RED = "#2a78d6", "#1baf7a", "#eda100", "#e34948"
ORDINAL_BLUES = ["#86b6ef", "#2a78d6", "#104281"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)

LOGS_DIR = os.path.join("experiments", "logs")
OUT_DIR = "figures"

CLASSES = [
    "archery", "driving car", "javelin throw", "passing American football (in game)",
    "playing drums", "playing guitar", "playing tennis", "pull ups", "scuba diving", "squat",
]
# Composizione del dataset nelle tre fasi (video per classe; conteggi dai
# riepiloghi di harvesting, fase 3 dopo la rimozione dei 6 video corrotti)
DATASET_PHASES = {
    "398 (originale)": [45, 38, 35, 35, 34, 43, 44, 42, 37, 45],
    "1.865 (+val/test)": [195, 175, 185, 185, 179, 189, 188, 191, 184, 194],
    "5.802 (+train)": [650, 583, 500, 518, 517, 608, 618, 612, 555, 641],
}
SCALE_LABELS = ["398", "1.865", "5.802"]
METRICS = ["purity", "nmi", "ari"]
METRIC_LABELS = {"purity": "Purity", "nmi": "NMI", "ari": "ARI"}
METRIC_COLORS = {"purity": BLUE, "nmi": AQUA, "ari": YELLOW}


def load_history(run_name: str) -> list[dict]:
    """Carica lo storico di un run dal suo ``history.json``."""
    with open(os.path.join(LOGS_DIR, run_name, "history.json"), encoding="utf-8") as f:
        return json.load(f)


def integer_x_axis(ax: plt.Axes) -> None:
    """Forza tick interi sull'asse x (le iterazioni sono conteggi discreti)."""
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Applica lo stile comune: sfondo, griglia leggera, assi discreti."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=11)


def new_figure(width: float, height: float) -> plt.Figure:
    """Crea una figura con superficie e margini coerenti."""
    fig = plt.figure(figsize=(width, height), facecolor=SURFACE)
    return fig


def save(fig: plt.Figure, name: str) -> None:
    """Salva la figura in ``figures/`` e la chiude."""
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"salvata {path}")


def fig_dataset_growth() -> None:
    """F1 — crescita del dataset per classe nelle tre fasi."""
    fig = new_figure(10, 6)
    ax = fig.add_subplot(111)
    style_axes(ax, grid_axis="x")

    short_names = [c.replace("passing American football (in game)", "passing football") for c in CLASSES]
    positions = range(len(CLASSES))
    bar_height = 0.26

    for i, (phase, counts) in enumerate(DATASET_PHASES.items()):
        offsets = [p + (i - 1) * bar_height for p in positions]
        ax.barh(offsets, counts, height=bar_height - 0.04, color=ORDINAL_BLUES[i], label=phase)

    ax.set_yticks(list(positions))
    ax.set_yticklabels(short_names, fontsize=11, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("video per classe", fontsize=12, color=INK2)
    ax.set_title(
        "Crescita del dataset — 10 classi di Kinetics-400, tre fasi di harvesting",
        fontsize=15, color=INK, fontweight="bold", pad=14,
    )
    # Legenda sotto il grafico: dentro l'area dati collide con le barre
    ax.legend(frameon=False, fontsize=11, loc="upper center",
              bbox_to_anchor=(0.5, -0.07), ncol=3, labelcolor=INK2)
    save(fig, "dataset_growth.png")


def fig_loop_schema() -> None:
    """F2 — schema a blocchi del loop iterativo."""
    fig = new_figure(10, 5.6)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.set_facecolor(SURFACE)

    def box(x, y, w, h, text, edge=BLUE, dashed=False):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12", linewidth=1.8,
            edgecolor=edge, facecolor="#ffffff", linestyle="--" if dashed else "-",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11.5, color=INK)

    def arrow(x1, y1, x2, y2, edge=INK2, dashed=False):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
            linewidth=1.8, color=edge, linestyle="--" if dashed else "-",
        ))

    # Ciclo principale (senso orario)
    box(0.4, 3.9, 2.9, 1.1, "Estrazione feature\n(backbone in eval)")
    box(4.0, 3.9, 2.9, 1.1, "K-Means su feature\nL2-normalizzate (K=10)")
    box(4.0, 0.6, 2.9, 1.1, "Pseudo-label\n(assegnazioni correnti)")
    box(0.4, 0.6, 2.9, 1.1, "Fine-tuning selettivo\n(testa nuova, CE pesata)")

    arrow(3.3, 4.45, 4.0, 4.45)
    arrow(5.45, 3.9, 5.45, 1.7)
    arrow(4.0, 1.15, 3.3, 1.15)
    arrow(1.85, 1.7, 1.85, 3.9)

    # Osservatori esterni al flusso di controllo
    box(7.6, 3.9, 2.2, 1.1, "Valutazione\npurity / NMI / ARI\n(solo osservazione)", edge=MUTED, dashed=True)
    box(7.6, 0.6, 2.2, 1.1, "Stop interno:\nstabilità NMI tra\nassegnazioni ≥ 0.95", edge=AQUA)
    arrow(6.9, 4.45, 7.6, 4.45, edge=MUTED, dashed=True)
    arrow(6.9, 1.15, 7.6, 1.15, edge=AQUA)

    ax.text(1.85, 5.35, "Iterazione 0 = baseline (nessun fine-tuning)", fontsize=10.5, color=INK2, ha="left")
    ax.set_title(
        "Clustering iterativo in stile DeepCluster — anatomia del loop",
        fontsize=15, color=INK, fontweight="bold", pad=12,
    )
    save(fig, "iterative_loop_schema.png")


def mean_loss(entry: dict) -> float | None:
    """Loss media dell'iterazione (None per l'iterazione 0)."""
    losses = entry["epoch_losses"]
    return sum(losses) / len(losses) if losses else None


def fig_resnet_regimes() -> None:
    """F3 — ResNet a 398 video: regime aggressivo vs gentile."""
    aggressive = load_history("iterative_resnet")
    gentle = load_history("iterative_resnet_gentle")

    fig = new_figure(11, 4.4)
    fig.suptitle(
        "ResNet18, 398 video — regime di fine-tuning aggressivo vs gentile",
        fontsize=15, color=INK, fontweight="bold",
    )

    ax1 = fig.add_subplot(121)
    style_axes(ax1)
    for hist, label, color in ((aggressive, "aggressivo (2 epoche, lr 1e-4)", RED),
                               (gentle, "gentile (1 epoca, lr 1e-5)", BLUE)):
        ax1.plot([e["iteration"] for e in hist], [e["purity"] * 100 for e in hist],
                 marker="o", markersize=6, linewidth=2, color=color, label=label)
    ax1.set_xlabel("iterazione", fontsize=12, color=INK2)
    ax1.set_ylabel("purity (%)", fontsize=12, color=INK2)
    ax1.set_title("Qualità dei cluster", fontsize=13, color=INK)
    ax1.legend(frameon=False, fontsize=10.5, labelcolor=INK2)

    ax2 = fig.add_subplot(122)
    style_axes(ax2)
    for hist, label, color in ((aggressive, "aggressivo", RED), (gentle, "gentile", BLUE)):
        iters = [e["iteration"] for e in hist if mean_loss(e) is not None]
        losses = [mean_loss(e) for e in hist if mean_loss(e) is not None]
        ax2.plot(iters, losses, marker="o", markersize=6, linewidth=2, color=color, label=label)
    ax2.set_xlabel("iterazione", fontsize=12, color=INK2)
    ax2.set_ylabel("loss media (cross-entropy pesata)", fontsize=12, color=INK2)
    ax2.set_title("Il termometro del regime: la loss", fontsize=13, color=INK)
    ax2.legend(frameon=False, fontsize=10.5, labelcolor=INK2)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "resnet398_regimes.png")


def fig_videomae_scaling_curves() -> None:
    """F4 — VideoMAE: metriche per iterazione alle tre scale."""
    runs = [
        ("iterative_videomae", "398 video"),
        ("iterative_videomae_n1865", "1.865 video"),
        ("iterative_videomae_lr3e6_n5802", "5.802 video (LR calibrato)"),
    ]

    fig = new_figure(12.5, 4.2)
    fig.suptitle(
        "VideoMAE — metriche per iterazione alle tre scale di dataset (K=10, 2 blocchi)",
        fontsize=15, color=INK, fontweight="bold",
    )

    legend_handles = None
    for m_index, metric in enumerate(METRICS):
        ax = fig.add_subplot(1, 3, m_index + 1)
        style_axes(ax)
        for r_index, (run, label) in enumerate(runs):
            hist = load_history(run)
            ax.plot([e["iteration"] for e in hist], [e[metric] * 100 for e in hist],
                    marker="o", markersize=5.5, linewidth=2,
                    color=ORDINAL_BLUES[r_index], label=label)
        integer_x_axis(ax)
        ax.set_xlabel("iterazione", fontsize=11.5, color=INK2)
        ax.set_ylabel(f"{METRIC_LABELS[metric]} (%)", fontsize=11.5, color=INK2)
        ax.set_title(METRIC_LABELS[metric], fontsize=13, color=INK)
        if legend_handles is None:
            legend_handles = ax.get_legend_handles_labels()

    # Legenda unica sotto la figura: dentro i pannelli coprirebbe le curve
    fig.legend(*legend_handles, frameon=False, fontsize=10.5, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 0.91))
    save(fig, "videomae_scaling.png")


def delta_points(hist: list[dict], metric: str) -> float:
    """Guadagno finale del run rispetto all'iterazione 0, in punti."""
    return (hist[-1][metric] - hist[0][metric]) * 100


def fig_scaling_deltas() -> None:
    """F5 — guadagni finali (Δ vs iterazione 0) alle tre scale."""
    backbone_runs = {
        "ResNet18 (regime gentile)": [
            "iterative_resnet_gentle", "iterative_resnet_gentle_n1865", "iterative_resnet_lr3e6_n5802",
        ],
        "VideoMAE (K=10, 2 blocchi)": [
            "iterative_videomae", "iterative_videomae_n1865", "iterative_videomae_lr3e6_n5802",
        ],
    }

    fig = new_figure(11.5, 4.4)
    fig.suptitle(
        "Guadagno del clustering iterativo per scala di dataset (iterazione finale − iterazione 0)",
        fontsize=15, color=INK, fontweight="bold",
    )

    legend_handles = None
    for b_index, (backbone, runs) in enumerate(backbone_runs.items()):
        ax = fig.add_subplot(1, 2, b_index + 1)
        style_axes(ax)
        histories = [load_history(run) for run in runs]

        width = 0.26
        all_values = []
        for m_index, metric in enumerate(METRICS):
            xs = [i + (m_index - 1) * width for i in range(len(SCALE_LABELS))]
            values = [delta_points(hist, metric) for hist in histories]
            all_values.extend(values)
            bars = ax.bar(xs, values, width=width - 0.03,
                          color=METRIC_COLORS[metric], label=f"Δ {METRIC_LABELS[metric]}")
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        value + (0.12 if value >= 0 else -0.40),
                        f"{value:+.1f}", ha="center", fontsize=9.5, color=INK)

        # Margini verticali ampi: le etichette dei valori devono respirare
        ax.set_ylim(min(0.0, min(all_values)) - 0.9, max(all_values) + 0.9)
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_xticks(range(len(SCALE_LABELS)))
        ax.set_xticklabels([f"{s} video" for s in SCALE_LABELS], fontsize=11, color=INK)
        ax.set_ylabel("Δ (punti percentuali)", fontsize=11.5, color=INK2)
        ax.set_title(backbone, fontsize=13, color=INK)
        if legend_handles is None:
            legend_handles = ax.get_legend_handles_labels()

    fig.legend(*legend_handles, frameon=False, fontsize=10.5, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    save(fig, "scaling_deltas.png")


def fig_stability() -> None:
    """F6 — stabilità (NMI tra assegnazioni consecutive) per iterazione."""
    backbone_runs = {
        "ResNet18": [
            ("iterative_resnet_gentle", "398 video"),
            ("iterative_resnet_gentle_n1865", "1.865 video"),
            ("iterative_resnet_lr3e6_n5802", "5.802 video (LR calibrato)"),
        ],
        "VideoMAE": [
            ("iterative_videomae", "398 video"),
            ("iterative_videomae_n1865", "1.865 video"),
            ("iterative_videomae_lr3e6_n5802", "5.802 video (LR calibrato)"),
        ],
    }

    fig = new_figure(11.5, 4.4)
    fig.suptitle(
        "Convergenza del loop — stabilità delle pseudo-label per iterazione",
        fontsize=15, color=INK, fontweight="bold",
    )

    for b_index, (backbone, runs) in enumerate(backbone_runs.items()):
        ax = fig.add_subplot(1, 2, b_index + 1)
        style_axes(ax)
        for r_index, (run, label) in enumerate(runs):
            hist = load_history(run)
            iters = [e["iteration"] for e in hist if e["stability"] is not None]
            stability = [e["stability"] for e in hist if e["stability"] is not None]
            ax.plot(iters, stability, marker="o", markersize=5.5, linewidth=2,
                    color=ORDINAL_BLUES[r_index], label=label)
        ax.axhline(0.95, color=MUTED, linewidth=1.4, linestyle="--")
        ax.text(0.6, 0.953, "soglia di stop (0.95)", fontsize=9.5, color=MUTED)
        ax.set_ylim(0.63, 1.0)
        integer_x_axis(ax)
        ax.set_xlabel("iterazione", fontsize=11.5, color=INK2)
        ax.set_ylabel("NMI tra assegnazioni consecutive", fontsize=11.5, color=INK2)
        ax.set_title(backbone, fontsize=13, color=INK)
        ax.legend(frameon=False, fontsize=10, labelcolor=INK2, loc="lower right")

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "stability_convergence.png")


def fig_ablations() -> None:
    """F7 — ablation su VideoMAE a 1.865 video (K e capacità)."""
    configs = [
        ("iterative_videomae_n1865", "Standard\n(K=10, 2 blocchi)"),
        ("iterative_videomae_k50_n1865", "Over-clustering\n(K=50)"),
        ("iterative_videomae_blocks4_n1865", "Più capacità\n(4 blocchi)"),
    ]

    fig = new_figure(8.5, 4.6)
    ax = fig.add_subplot(111)
    style_axes(ax)

    width = 0.32
    for m_index, metric in enumerate(("nmi", "ari")):
        xs = [i + (m_index - 0.5) * width for i in range(len(configs))]
        values = [delta_points(load_history(run), metric) for run, _ in configs]
        color = BLUE if metric == "nmi" else AQUA
        bars = ax.bar(xs, values, width=width - 0.04, color=color, label=f"Δ {METRIC_LABELS[metric]}")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.07,
                    f"{value:+.1f}", ha="center", fontsize=10.5, color=INK)

    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([label for _, label in configs], fontsize=11, color=INK)
    ax.set_ylabel("Δ finale (punti percentuali)", fontsize=11.5, color=INK2)
    ax.set_title(
        "Ablation su VideoMAE (1.865 video): la configurazione standard resta la migliore",
        fontsize=14, color=INK, fontweight="bold", pad=14,
    )
    ax.legend(frameon=False, fontsize=11, labelcolor=INK2)
    save(fig, "ablations_videomae.png")


def main() -> None:
    """Genera tutte le figure del report."""
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 12})

    fig_dataset_growth()
    fig_loop_schema()
    fig_resnet_regimes()
    fig_videomae_scaling_curves()
    fig_scaling_deltas()
    fig_stability()
    fig_ablations()


if __name__ == "__main__":
    main()
