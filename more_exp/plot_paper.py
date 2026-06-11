from pathlib import Path


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch




STYLE_BASE = "#E9CFA7"
STYLE_ACCENT = "#2A9D8F"
STYLE_ACCENT_DARK = "#1F766C"
STYLE_THIRD = "#8E79B9"
STYLE_THIRD_DARK = "#5B4A86"
STYLE_TEXT = "#233142"
STYLE_BASE_TEXT = "#8A5A20"
STYLE_GRID = "#D9DDE3"


def set_paper_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#FBFBFB",
            "font.size": 8,
            "grid.color": STYLE_GRID,
            "grid.linestyle": "--",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.55,
            "hatch.linewidth": 0.55,
        },
    )

def fig1(
    output_path: str = "exp_runs/save_imgs/fig1.pdf",
    png_output_path: str | None = None,
    figsize=(5.4, 2.65),
) -> Path:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    groups = [
        ("DirectInject", "GPT-5.4\nmini", 100.0, 17.8),
        ("DirectInject", "Claude Haiku\n4.5", 98.2, 10.7),
        ("AgentDojo", "GPT-5.4\nmini", 35 / 42 * 100, 13 / 42 * 100),
        ("AgentDojo", "Claude Haiku\n4.5", 41 / 42 * 100, 4 / 42 * 100),
    ]

    label_fs = 11
    value_fs = label_fs - 2
    set_paper_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    x = np.array([0, 1, 3, 4], dtype=float)
    text_values = np.array([row[2] for row in groups], dtype=float)
    defense_values = np.array([row[3] for row in groups], dtype=float)
    model_labels = [row[1] for row in groups]
    bar_w = 0.32

    bars_text = ax.bar(
        x - bar_w / 2,
        text_values,
        bar_w,
        label="Text",
        color=STYLE_BASE,
        edgecolor="white",
        linewidth=1.1,
        zorder=3,
    )
    bars_defense = ax.bar(
        x + bar_w / 2,
        defense_values,
        bar_w,
        label="Image",
        color=STYLE_ACCENT,
        edgecolor="white",
        linewidth=1.1,
        zorder=3,
    )

    for bars, color in ((bars_text, STYLE_BASE_TEXT), (bars_defense, STYLE_ACCENT_DARK)):
        for bar in bars:
            value = bar.get_height()
            value_label = "0" if np.isclose(value, 0.0) else f"{value:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 2.4,
                value_label,
                ha="center",
                va="bottom",
                fontsize=value_fs,
                fontweight="semibold",
                color=color,
                clip_on=False,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=label_fs)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=label_fs)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.grid(axis="y", linestyle="--", alpha=0.55)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=label_fs)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=2,
        frameon=False,
        fontsize=label_fs+1,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    sns.despine(ax=ax)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    benchmark_labels = [
        ("DirectInject", (x[0] + x[1]) / 2),
        ("AgentDojo", (x[2] + x[3]) / 2),
    ]
    for label, x_center in benchmark_labels:
        ax.text(
            x_center,
            -0.28,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=label_fs,
            fontweight="semibold",
        )
    fig.savefig(output, bbox_inches="tight", pad_inches=0.12)
    if png_output_path is not None:
        png_output = Path(png_output_path)
        png_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_output, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return output



def figx(
    output_path: str = "exp_runs/save_imgs/fig1.pdf",
    png_output_path: str | None = None,
    figsize=(4.6, 3.25),
) -> Path:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    data = {
        "Image": [
            ("GPT-5.4 mini", 75.0, 0),
            ("Claude Haiku 4.5", 17.9, 0),
        ],
        "Audio": [
            ("Qwen3.5 Omni Plus", 100.0, 55.4),
            ("GPT-Audio Mini", 78.6, 8.9),
        ],
    }

    set_paper_style()
    fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=300, sharey=True)
    bar_w = 0.32
    for ax, (panel, rows) in zip(axes, data.items()):
        labels = [row[0] for row in rows]
        text_values = np.array([row[1] for row in rows], dtype=float)
        defense_values = np.array([row[2] for row in rows], dtype=float)
        x = np.arange(len(labels))

        bars_text = ax.bar(
            x - bar_w / 2,
            text_values,
            bar_w,
            label="Text",
            color=STYLE_BASE,
            edgecolor="white",
            linewidth=1.1,
            zorder=3,
        )
        bars_defense = ax.bar(
            x + bar_w / 2,
            defense_values,
            bar_w,
            label="Image/Audio",
            color=STYLE_ACCENT,
            edgecolor="white",
            linewidth=1.1,
            zorder=3,
        )

        for bars, color in ((bars_text, STYLE_BASE_TEXT), (bars_defense, STYLE_ACCENT_DARK)):
            for bar in bars:
                value = bar.get_height()
                value_label = "0" if np.isclose(value, 0.0) else f"{value:.1f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 2.4,
                    value_label,
                    ha="center",
                    va="bottom",
                    fontsize=8.3,
                    fontweight="semibold",
                    color=color,
                    clip_on=False,
                )

        ax.text(
            0.98,
            0.94,
            panel,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.3,
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.23", facecolor="white", edgecolor="#C9CED6", linewidth=0.6),
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.3)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="y", labelsize=7.4)
        ax.tick_params(axis="x", length=0, pad=5)
        ax.grid(axis="y", linestyle="--", alpha=0.55)
        ax.grid(axis="x", visible=False)
        sns.despine(ax=ax)

    fig.supylabel("Attack Success Rate (%)", fontsize=8.3, x=0.075)
    axes[0].legend(
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=11,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.8,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.03, 0, 1, 0.94], h_pad=0.55)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.12)
    if png_output_path is not None:
        png_output = Path(png_output_path)
        png_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_output, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return output


def ablation_rendering(
    output_path: str = "exp_runs/save_imgs/ablation_rendering.pdf",
    figsize=(4.6, 2.85),
) -> Path:
    label_fs = 8.8
    value_fs = 7.2
    models = [
        "google/Gemini 3.1 Flash",
        "x-ai/Grok 4.3",
        "qwen/Qwen3.6-Plus",
        "qwen/Qwen3.6-Flash",
        "moonshotai/Kimi-K2.6",
    ]
    plain = np.array([19.1, 15.3, 17.3, 5.9, 0.3])
    google = np.array([3.3, 8.9, 7.7, 1.8, 0.0])
    blackboard = np.array([6.9, 10.2, 10.7, 1.8, 0.3])

    # Rank models by descending plain-image ASR for a clearer leaderboard view.
    order = np.argsort(-plain)
    models = [models[i] for i in order]
    plain = plain[order]
    google = google[order]
    blackboard = blackboard[order]
    model_labels = []
    for name in models:
        short_name = name.split("/", 1)[-1]
        model_labels.append(short_name)

    y = np.arange(len(models))
    bar_height = 0.22

    set_paper_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    bars_plain = ax.barh(
        y - bar_height,
        plain,
        bar_height,
        label="Plain",
        color=STYLE_BASE,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )
    bars_blackboard = ax.barh(
        y,
        blackboard,
        bar_height,
        label="Blackboard",
        color=STYLE_ACCENT,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )
    bars_google = ax.barh(
        y + bar_height,
        google,
        bar_height,
        label="Google search",
        color=STYLE_THIRD,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )

    for bars, color in (
        (bars_plain, STYLE_BASE_TEXT),
        (bars_blackboard, STYLE_ACCENT_DARK),
        (bars_google, STYLE_THIRD_DARK),
    ):
        for bar in bars:
            value = bar.get_width()
            value_label = "0" if np.isclose(value, 0.0) else f"{value:.1f}"
            ax.text(
                value + 0.45,
                bar.get_y() + bar.get_height() / 2,
                value_label,
                va="center",
                ha="left",
                fontsize=value_fs,
                fontweight="semibold",
                color=color,
                clip_on=False,
            )

    ax.set_xlim(0, 22.5)
    ax.set_xlabel("Attack Success Rate (%)", fontsize=label_fs)
    ax.set_yticks(y)
    ax.set_yticklabels(model_labels, fontsize=label_fs)
    ax.tick_params(axis="x", labelsize=label_fs)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linestyle="--", alpha=0.55)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    sns.despine(ax=ax, left=True)
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=8.8,
        handlelength=1.05,
        handletextpad=0.45,
        labelspacing=0.32,
        borderaxespad=0.35,
    )


    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def after_img_fine_tuned(
    models=("Gemma-4-E4B", "Qwen-3.5-4B", "Qwen3-Omni-30B"),
    base=(0.3, 8.9, 37.0),
    ft=(18.9, 21.9, 50.5),
    base_label="Before finetuning",
    ft_label="Increase after finetuning",
    c_base=STYLE_BASE,
    c_ft=STYLE_ACCENT,
    figsize=(4.6, 2.55),
    output_path: str = "exp_runs/save_imgs/after_img_fine_tuned.pdf",
) -> Path:
    """Grouped bar chart of strict ASR: base vs fine-tuned, per model.

    The stacked bars show the before-finetuning ASR plus the increase after finetuning.
    """
    base_values = np.array(base, dtype=float)
    ft_values = np.array(ft, dtype=float)
    gains = ft_values - base_values
    x = np.arange(len(models))
    width = 0.58

    set_paper_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=170)

    bars_base = ax.bar(
        x,
        base_values,
        width,
        label=base_label,
        color=c_base,
        edgecolor="white",
        linewidth=1.4,
        zorder=3,
    )
    bars_gain = ax.bar(
        x,
        gains,
        width,
        bottom=base_values,
        label=ft_label,
        color=c_ft,
        edgecolor="white",
        linewidth=1.4,
        zorder=3,
    )

    for idx, (before, after, gain) in enumerate(zip(base_values, ft_values, gains)):
        ax.text(
            x[idx],
            after + max(ft_values) * 0.025,
            f"{after:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#233142",
        )
        ax.text(
            x[idx],
            before + gain / 2,
            f"+{gain:.1f}",
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="white",
        )
        if before < 5:
            ax.annotate(
                f"{before:.1f}",
                xy=(x[idx], before / 2),
                xytext=(x[idx] + width * 0.62, before + max(ft_values) * 0.035),
                textcoords="data",
                ha="left",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color="#8A5A20",
                bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor=c_base, alpha=0.95),
                arrowprops=dict(arrowstyle="-", color=c_base, linewidth=0.8),
                zorder=5,
            )
        else:
            ax.text(
                x[idx],
                before / 2,
                f"{before:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="#8A5A20",
                zorder=5,
            )

    ax.set_xlabel("")
    ax.set_ylabel("Attack Success Rate (%)", fontsize=8.5)
    ax.set_ylim(0, max(ft_values) * 1.25)
    ax.set_yticks(np.arange(0, 61, 20))
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.tick_params(axis="x", labelsize=7.0, pad=2)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)
    ax.legend(
        title="",
        loc="upper left",
        fontsize=9,
        frameon=False,
        fancybox=True,
        borderpad=0.3,
        handlelength=1.2,
        handletextpad=0.55,
        labelspacing=0.4,
        borderaxespad=0.35,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def base_vs_instruct(
    output_path: str = "exp_runs/save_imgs/base_vs_instruct.pdf",
    figsize=(4.6, 3.2),
    colors=None,
) -> Path:
    """Grouped bar chart: ASR increase after instruct-tuning, one panel per model family."""
    data = {
        "Qwen3.5": {
            "-0.8B": {"text": (9.5, 26.2), "image": (0.0, 11.9)},
            "-9B": {"text": (2.4, 69.0), "image": (0.0, 4.8)},
        },
        "InternVL3.5": {
            "-1B": {"text": (14.3, 90.5), "image": (0.0, 0.0)},
            "-38B": {"text": (50.0, 57.1), "image": (2.4, 4.8)},
        },
    }

    colors = colors or {"text": STYLE_BASE, "image": STYLE_ACCENT}
    label_colors = {"text": "#A2671A", "image": STYLE_ACCENT_DARK}
    families = list(data.keys())
    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#FBFBFB",
            "grid.color": "#D9DDE3",
            "grid.linewidth": 0.45,
            "hatch.linewidth": 0.55,
        },
    )
    fig, axes = plt.subplots(len(families), 1, figsize=figsize, dpi=300, sharex=False, sharey=True)
    axes = np.atleast_1d(axes).tolist()

    bar_w = 0.28
    channel_offsets = {"text": -bar_w * 0.57, "image": bar_w * 0.57}
    max_gain = max(
        data[family][size][channel][1] - data[family][size][channel][0]
        for family in families
        for size in data[family]
        for channel in ("text", "image")
    )
    y_max = max(20, np.ceil((max_gain + 6) / 10) * 10)

    for ax, family in zip(axes, families):
        sizes = list(data[family].keys())
        x = np.arange(len(sizes))

        for channel in ("text", "image"):
            base_values = np.array([data[family][s][channel][0] for s in sizes], dtype=float)
            instruct_values = np.array([data[family][s][channel][1] for s in sizes], dtype=float)
            gains = instruct_values - base_values
            xpos = x + channel_offsets[channel]
            color = colors[channel]
            label_color = label_colors[channel]

            ax.bar(
                xpos,
                gains,
                width=bar_w,
                facecolor=color,
                edgecolor="white",
                linewidth=1.0,
                zorder=2,
            )

            for xpos_i, gain in zip(xpos, gains):
                ax.annotate(
                    f"+{gain:g}",
                    (xpos_i, gain),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.8,
                    color=label_color,
                    fontweight="semibold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels([f"{family}{size}" for size in sizes], fontsize=8.2)
        ax.set_ylim(0, y_max)
        ax.set_yticks(np.arange(0, y_max + 1, 20))
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}")
        ax.tick_params(axis="y", labelsize=7.6)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.grid(axis="y", linestyle="--", alpha=0.75)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

    fig.supylabel(r"$\Delta$ ASR After Instruction Tuning (%)", fontsize=8.8, x=0.075)

    input_handles = [
        Patch(facecolor=colors["text"], edgecolor="white", label="Text"),
        Patch(facecolor=colors["image"], edgecolor="white", label="Image"),
    ]
    axes[0].legend(
        handles=input_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=2,
        frameon=False,
        fancybox=True,
        fontsize=9,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.65,
        borderpad=0.25,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.03, 0, 1, 1], h_pad=0.45)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output


def agent_redteaming(
    output_path: str = "exp_runs/save_imgs/agent_redteaming.pdf",
    figsize: tuple[float, float] = (5, 4),
) -> Path:
    """Render ASR bars for text, image, and text→image transfer and save the figure."""
    data = {
        "DirectInject": {
            "GPT-5.4 mini": (98.2, 7.1, 0.0),
            # "gemini-3.1-flash": (100.0, 94.6, 80.4),
            "Kimi K2.6": (85.7, 48.2, 50.0),
            "Claude Haiku 4.5": (96.4, 10.7, 0.0),
        },
        "AgentDojo": {
            "GPT-5.4 mini": (40.05, 4.0, 21.3),
            # "gemini-3.1-flash": (92, 30.7, 56.0),
            "Kimi K2.6": (45.3, 20.0, 12.0),
            "Claude Haiku 4.5": (10.7, 1.3, 0.0),
        },
    }

    text_color = STYLE_BASE
    transfer_color = STYLE_ACCENT
    image_color = STYLE_THIRD
    label_fs = 10
    value_fs = 8

    benchmarks = list(data.keys())
    bar_width = 0.2
    group_offset = bar_width + 0.025
    tick_step = 50
    y_max = 106

    set_paper_style()
    fig, axes = plt.subplots(len(benchmarks), 1, figsize=figsize, dpi=300, sharex=True, sharey=True)
    if len(benchmarks) == 1:
        axes = [axes]

    for ax, benchmark in zip(axes, benchmarks):
        models = list(data[benchmark].keys())
        x = np.arange(len(models))
        text_values = np.array([data[benchmark][model][0] for model in models])
        image_values = np.array([data[benchmark][model][1] for model in models])
        transfer_values = np.array([data[benchmark][model][2] for model in models])

        bars_text = ax.bar(
            x - group_offset,
            text_values,
            bar_width,
            label="Text",
            color=text_color,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        bars_transfer = ax.bar(
            x,
            transfer_values,
            bar_width,
            label="Transfer (text→image)",
            color=transfer_color,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        bars_image = ax.bar(
            x + group_offset,
            image_values,
            bar_width,
            label="Image",
            color=image_color,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )

        max_value = float(max(text_values.max(), image_values.max(), transfer_values.max()))
        label_padding = max(1.0, max_value * 0.02)
        for series_idx, (bars, color) in enumerate(
            (
                (bars_text, STYLE_BASE_TEXT),
                (bars_transfer, STYLE_ACCENT_DARK),
                (bars_image, STYLE_THIRD_DARK),
            )
        ):
            for bar in bars:
                value = bar.get_height()
                y_offset = label_padding + series_idx * label_padding * 0.2
                value_label = "0" if np.isclose(value, 0.0) else f"{value:.1f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + y_offset,
                    value_label,
                    ha="center",
                    va="bottom",
                    fontsize=value_fs,
                    fontweight="semibold",
                    color=color,
                    clip_on=False,
                )

        ax.set_title(benchmark, fontsize=label_fs + 0.4, fontweight="semibold", pad=5)
        ax.set_ylim(0, y_max)
        ax.set_yticks(np.arange(0, y_max + 1, tick_step))
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=label_fs)
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}")
        ax.tick_params(axis="y", labelsize=label_fs)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.grid(axis="y", linestyle="--", alpha=0.55)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

    legend_ax = axes[1] if len(axes) > 1 else axes[0]
    legend_ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.98),
        ncol=3,
        frameon=False,
        fontsize=label_fs,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.75,
    )
    fig.supylabel("Attack Success Rate (%)", fontsize=label_fs, x=0.075)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.03, 0, 1, 0.94], h_pad=0.75)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def text_vs_image(
    output_path: str = "exp_runs/save_imgs/text_vs_image.pdf",
    png_output_path: str | None = None,
    figsize=(6.8, 4.9),
) -> Path:
    """Figure for Table 1: text-only / image-only / conflict compliance."""
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    models = [
        "GPT-5.4\nnano",
        "Claude Haiku\n4.5",
        "Grok 4.3",
        "Gemini 3.1\nFlash",
        "Kimi\nK2.6",
        "Qwen3.6\nPlus",
    ]

    text_only = [99.5, 98.1, 100.0, 99.0, 97.6, 97.6]
    text_only_err = [1.1, 2.0, 0.0, 2.1, 1.7, 2.4]
    image_only = [100.0, 93.8, 100.0, 94.3, 99.0, 90.0]
    image_only_err = [0.0, 2.1, 0.0, 1.3, 1.3, 2.0]

    text_wins = [85.2, 66.2, 88.3, 82.6, 99.7, 92.9]
    text_wins_err = [3.2, 1.4, 2.1, 1.8, 0.6, 2.1]
    image_wins = [14.8, 32.4, 4.5, 17.1, 0.3, 6.9]
    image_wins_err = [3.2, 1.6, 2.1, 1.6, 0.6, 2.3]

    label_fs = 11.5
    set_paper_style()
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=figsize,
        dpi=300,
        sharey=True,
    )

    x = np.arange(len(models))
    width = 0.3

    ax1.bar(
        x - width / 2,
        text_only,
        width,
        yerr=text_only_err,
        label="Text",
        color=STYLE_BASE,
        edgecolor="white",
        linewidth=1.1,
        capsize=2.4,
        error_kw={"elinewidth": 0.65, "ecolor": STYLE_TEXT},
        zorder=3,
    )
    ax1.bar(
        x + width / 2,
        image_only,
        width,
        yerr=image_only_err,
        label="Image",
        color=STYLE_ACCENT,
        edgecolor="white",
        linewidth=1.1,
        capsize=2.4,
        error_kw={"elinewidth": 0.65, "ecolor": STYLE_TEXT},
        zorder=3,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=label_fs)

    ax2.bar(
        x - width / 2,
        text_wins,
        width,
        yerr=text_wins_err,
        label="Text",
        color=STYLE_BASE,
        edgecolor="white",
        linewidth=1.1,
        capsize=2.4,
        error_kw={"elinewidth": 0.65, "ecolor": STYLE_TEXT},
        zorder=3,
    )
    ax2.bar(
        x + width / 2,
        image_wins,
        width,
        yerr=image_wins_err,
        label="Image",
        color=STYLE_ACCENT,
        edgecolor="white",
        linewidth=1.1,
        capsize=2.4,
        error_kw={"elinewidth": 0.65, "ecolor": STYLE_TEXT},
        zorder=3,
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=label_fs)

    for ax in (ax1, ax2):
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="y", labelsize=label_fs)
        ax.tick_params(axis="x", length=0, pad=5)
        ax.grid(axis="y", linestyle="--", alpha=0.55)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

    fig.supylabel("Instruction-following Rate (%)", fontsize=label_fs + 1, x=0.045)
    legend_handles = [
        Patch(facecolor=STYLE_BASE, edgecolor="white", label="Text"),
        Patch(facecolor=STYLE_ACCENT, edgecolor="white", label="Image"),
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.03, 0.04, 1, 0.9], h_pad=3.8)
    title_offset = 0.095
    for ax, title in (
        (ax1, "(a) Single-modality"),
        (ax2, "(b) Conflict: text vs. image"),
    ):
        bbox = ax.get_position()
        fig.text(
            (bbox.x0 + bbox.x1) / 2,
            bbox.y0 - title_offset,
            title,
            ha="center",
            va="top",
            fontsize=label_fs + 0.3,
            fontweight="semibold",
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
        fontsize=label_fs + 1,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.65,
    )
    fig.savefig(output, bbox_inches="tight", pad_inches=0.12)
    if png_output_path is not None:
        png_output = Path(png_output_path)
        png_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_output, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return output


def human_redteaming(
    output_path: str = "exp_runs/save_imgs/human_redteaming.pdf",
    png_output_path: str | None = None,
    figsize=(4.6, 2.65),
) -> Path:
    """Plot human red-teaming ASR in the same style as fig1."""
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    models = ["Claude\nHaiku 4.5", "GPT-5.4\nmini", "GPT-5.5", "Claude\nOpus 4.7"]
    text_asr = np.array([97.6, 47.6, 19.0, 14.3])
    image_asr = np.array([7.1, 21.4, 16.7, 11.9])

    label_fs = 10
    value_fs = 8
    set_paper_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    x = np.arange(len(models))
    width = 0.32
    bars_text = ax.bar(
        x - width / 2,
        text_asr,
        width,
        label="Text",
        color=STYLE_BASE,
        edgecolor="white",
        linewidth=1.1,
        zorder=3,
    )
    bars_image = ax.bar(
        x + width / 2,
        image_asr,
        width,
        label="Image",
        color=STYLE_ACCENT,
        edgecolor="white",
        linewidth=1.1,
        zorder=3,
    )

    for bars, color in ((bars_text, STYLE_BASE_TEXT), (bars_image, STYLE_ACCENT_DARK)):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 2.0,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=value_fs,
                fontweight="semibold",
                color=color,
                clip_on=False,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=label_fs)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=label_fs)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.grid(axis="y", linestyle="--", alpha=0.55)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=label_fs)
    ax.legend(
        loc="upper right",
        ncol=2,
        frameon=False,
        fontsize=11,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    sns.despine(ax=ax)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", pad_inches=0.12)
    if png_output_path is not None:
        png_output = Path(png_output_path)
        png_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_output, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return output


# ==================== Not used for now ============================

def plot_audio(
    output_path: str = "exp_runs/save_imgs/audio_headline_results.pdf",
    figsize=(4.6, 2.65),
) -> Path:
    models = ["GPT-Audio Mini", "Qwen3.5 Omni Plus"]
    conds = ["Text", "Audio"]
    markers = {"Text": "o", "Audio": "s"}
    model_colors = {"GPT-Audio Mini": STYLE_BASE, "Qwen3.5 Omni Plus": STYLE_ACCENT}

    asr = np.array([[27.3, 1.3], [50.5, 15.8]])
    ut = np.array([[100.0, 91.3], [82.1, 93.4]])

    label_fs = 9.0
    set_paper_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    label_offsets = {
        ("GPT-Audio Mini", "Text"): (-8, 7),
        ("GPT-Audio Mini", "Audio"): (7, 7),
        ("Qwen3.5 Omni Plus", "Text"): (7, 7),
        ("Qwen3.5 Omni Plus", "Audio"): (7, 7),
    }

    for model_idx, model in enumerate(models):
        for cond_idx, cond in enumerate(conds):
            x_val = ut[model_idx, cond_idx]
            y_val = asr[model_idx, cond_idx]
            ax.scatter(
                x_val,
                y_val,
                s=95,
                marker=markers[cond],
                color=model_colors[model],
                edgecolor="white",
                linewidth=1.1,
                zorder=5,
            )
            dx, dy = label_offsets[(model, cond)]
            ax.annotate(
                cond,
                xy=(x_val, y_val),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=label_fs - 1.0,
                ha="left" if dx > 0 else "right",
                va="center",
                color=model_colors[model],
                fontweight="semibold",
            )

    ax.axvline(100.0, color="#777777", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.text(
        99.0,
        54,
        "No-attack utility",
        ha="right",
        va="top",
        fontsize=label_fs - 1.1,
        color="#666666",
    )
    ax.set_xlim(78, 104)
    ax.set_ylim(-4, 58)
    ax.set_xticks([80, 85, 90, 95, 100])
    ax.set_yticks([0, 20, 40])
    ax.tick_params(axis="both", labelsize=label_fs - 0.7)
    ax.grid(axis="both", linestyle="--", alpha=0.55)
    ax.set_xlabel("Utility Under Attack (%)", fontsize=label_fs)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=label_fs)
    sns.despine(ax=ax)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=model_colors["GPT-Audio Mini"],
               markeredgecolor="white", markersize=6, label="GPT-Audio Mini"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=model_colors["Qwen3.5 Omni Plus"],
               markeredgecolor="white", markersize=6, label="Qwen3.5 Omni Plus"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
        fontsize=label_fs - 0.7,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.75,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


    
if __name__ == "__main__":
    for plot_fn in (
        fig1,
        ablation_rendering,
        after_img_fine_tuned,
        base_vs_instruct,
        agent_redteaming,
        plot_audio,
    ):
        path = plot_fn()
        print(f"Saved figure to {path}")
