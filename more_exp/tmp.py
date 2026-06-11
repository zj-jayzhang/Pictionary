from pathlib import Path


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch


ASR_BARS_DATA = {
    "Qwen3.5": {
        "0.8B": {"text": (9.5, 26.2), "image": (0.0, 11.9)},
        "9B": {"text": (2.4, 69.0), "image": (0.0, 4.8)},
    },
    "InternVL3.5": {
        "1B": {"text": (14.3, 90.5), "image": (0.0, 0.0)},
        "38B": {"text": (50.0, 57.1), "image": (2.4, 4.8)},
    },
}


def fig1(
    output_path: str = "exp_runs/save_imgs/asr_chart.pdf",
    png_output_path: str | None = "exp_runs/save_imgs/asr_chart.png",
) -> Path:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    # (model, modality, text-ASR, defense-ASR)
    data = [
        ("gpt-5.4-mini", "Image", 75.0, 0.0),
        ("claude-haiku-4.5", "Image", 17.9, 0.0),
        ("gpt-audio-mini", "Audio", 78.6, 8.9),
        ("qwen3.5-omni-plus", "Audio", 100.0, 55.4),
    ]

    red = "#E2655F"
    red_edge = "#A32D2D"
    green = "#8AC247"
    green_edge = "#3B6D11"
    teal = "#2DB7A0"
    teal_edge = "#0F6E56"
    arrow_yellow = "#E6A817"
    arrow_blue = "#185FA5"
    arrow_lw = 1.8
    tick_gray = "#5F5E5A"
    grid_gray = "#B4B2A9"

    fig, ax = plt.subplots(figsize=(9.0, 3), dpi=170)

    bar_w = 0.36
    pair_gap = 0.04
    group_gap = 0.55
    modality_gap = 0.95

    pos_t = []
    pos_d = []
    x = 0.0
    for i, _ in enumerate(data):
        if i == 2:
            x += modality_gap
        elif i > 0:
            x += group_gap
        pos_t.append(x)
        pos_d.append(x + bar_w + pair_gap)
        x += bar_w * 2 + pair_gap

    for i, (name, modality, text_asr, defense_asr) in enumerate(data):
        is_image = modality == "Image"
        defense_fill = green if is_image else teal
        defense_edge = green_edge if is_image else teal_edge

        ax.bar(pos_t[i], text_asr, bar_w, color=red, edgecolor=red_edge, linewidth=0.7, zorder=3)
        ax.bar(
            pos_d[i],
            defense_asr,
            bar_w,
            color=defense_fill,
            edgecolor=defense_edge,
            linewidth=0.7,
            zorder=3,
        )

        ax.text(
            pos_t[i],
            text_asr + 2.8,
            f"{text_asr:.1f}",
            ha="center",
            fontsize=10.5,
            fontweight="bold",
            color="#2C2C2A",
            zorder=5,
        )
        defense_label = f"{defense_asr:.1f}" if defense_asr > 0 else "0.0"
        ax.text(
            pos_d[i],
            max(defense_asr, 0) + 2.8,
            defense_label,
            ha="center",
            fontsize=10.5,
            fontweight="bold",
            color="#2C2C2A",
            zorder=5,
        )

        if text_asr > 0:
            defense_label_y = max(defense_asr, 0) + 2.8
            arrow = FancyArrowPatch(
                (pos_t[i] + bar_w / 2, text_asr),
                (pos_d[i], defense_label_y + 5),
                arrowstyle="->",
                mutation_scale=15,
                linestyle="-",
                linewidth=arrow_lw,
                color=arrow_yellow if is_image else arrow_blue,
                zorder=4,
            )
            ax.add_patch(arrow)

        pair_center = (pos_t[i] + pos_d[i]) / 2
        ax.text(pair_center, -6, name, ha="center", va="top", fontsize=10, color="#2C2C2A")

    bracket_y = -16
    for left, right, label, color in [
        (pos_t[0] - bar_w * 0.2, pos_d[1] + bar_w * 0.2, "Image defense", green_edge),
        (pos_t[2] - bar_w * 0.2, pos_d[3] + bar_w * 0.2, "Audio defense", teal_edge),
    ]:
        ax.plot([left, right], [bracket_y, bracket_y], color=color, linewidth=1.1, clip_on=False)
        ax.plot([left, left], [bracket_y, bracket_y + 1.8], color=color, linewidth=1.1, clip_on=False)
        ax.plot([right, right], [bracket_y, bracket_y + 1.8], color=color, linewidth=1.1, clip_on=False)
        ax.text(
            (left + right) / 2,
            bracket_y - 4,
            label,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=color,
        )

    ax.set_ylim(-26, 118)
    ax.set_xlim(pos_t[0] - 0.45, pos_d[-1] + 0.45)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=10.5, color="#2C2C2A")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=9.5, colors=tick_gray, length=3)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color(grid_gray)
    ax.spines["left"].set_linewidth(0.6)
    ax.grid(axis="y", linestyle=(0, (1.5, 3)), linewidth=0.5, color=grid_gray, alpha=0.6, zorder=1)
    ax.set_axisbelow(True)

    legend_elements = [
        Line2D(
            [0, 1],
            [0, 0],
            color=arrow_yellow,
            lw=arrow_lw,
            linestyle="-",
            marker=">",
            markevery=[1],
            markersize=9,
            markerfacecolor=arrow_yellow,
            markeredgecolor=arrow_yellow,
            label="Text to Image",
        ),
        Line2D(
            [0, 1],
            [0, 0],
            color=arrow_blue,
            lw=arrow_lw,
            linestyle="-",
            marker=">",
            markevery=[1],
            markersize=9,
            markerfacecolor=arrow_blue,
            markeredgecolor=arrow_blue,
            label="Text to Audio",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=10,
        handlelength=3.0,
        handletextpad=0.7,
        columnspacing=3.0,
    )

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


def ablation_rendering(output_path: str = "exp_runs/save_imgs/ablation_rendering.png") -> Path:
    font_size = 18
    bar_value_font_size = 18
    models = [
        "google/gemini-3.1-flash",
        "x-ai/grok-4.3",
        "qwen/qwen3.6-plus",
        "qwen/qwen3.6-flash",
        "moonshotai/kimi-k2.6",
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
        if short_name.startswith("gemini-3.1"):
            short_name = "gemini-3.1-flash"
        model_labels.append(short_name)

    x = np.arange(len(models))
    bar_width = 0.22

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=170)

    bars_plain = ax.bar(x - bar_width, plain, bar_width, label="Plain", color="#4C78A8")
    bars_blackboard = ax.bar(x, blackboard, bar_width, label="Blackboard", color="#F58518")
    bars_google = ax.bar(x + bar_width, google, bar_width, label="Google search", color="#72B7B2")

    max_value = float(np.max([plain.max(), blackboard.max(), google.max()]))
    label_padding = max(0.45, max_value * 0.03)

    extra_label_offsets = {
        (1, 1): (bar_width * 0.35, 0.0),  # 10.7%
        (1, 2): (bar_width * 0.35, 0.0),  # 10.2%
        (2, 3): (bar_width * 0.35, label_padding * 0.75),  # second 1.8%
        (1, 4): (bar_width * 0.35, label_padding * 0.75),  # second 0.3%
        (2, 4): (bar_width * 0.35, 0.3 - 2 * label_padding * 0.55),  # align 0 with first 0.3%
    }

    for series_idx, bars in enumerate((bars_plain, bars_blackboard, bars_google)):
        for model_idx, bar in enumerate(bars):
            value = bar.get_height()
            value_label = "0" if np.isclose(value, 0.0) else f"{value:.1f}%"
            # Stagger labels by series to avoid overlap on small/close values.
            series_lift = series_idx * label_padding * 0.55
            extra_x, extra_y = extra_label_offsets.get((series_idx, model_idx), (0.0, 0.0))
            ax.text(
                bar.get_x() + bar.get_width() / 2 + bar.get_width() * 0.24 + extra_x,
                value + label_padding + series_lift + extra_y,
                value_label,
                va="bottom",
                ha="center",
                fontsize=bar_value_font_size,
                clip_on=False,
            )

    ax.set_ylim(0, max_value + label_padding * 3.0)
    # ax.set_xlabel("Model", fontsize=font_size)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=font_size)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.legend(loc="upper right", frameon=True, fontsize=font_size)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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
    c_base="#E9CFA7",
    c_ft="#2A9D8F",
    figsize=(3.45, 2.55),
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

    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 8,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
        },
    )
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
    ax.set_ylabel("ASR (%)", fontsize=8.5)
    ax.set_ylim(0, max(ft_values) * 1.25)
    ax.set_xticks(x)
    ax.set_xticklabels([model.replace("-", "-\n", 1) for model in models])
    ax.tick_params(axis="x", labelsize=7.5, pad=2)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)
    ax.legend(
        title="",
        loc="upper left",
        fontsize=8.5,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        borderpad=0.3,
        handlelength=1.1,
        handletextpad=0.4,
        labelspacing=0.25,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def base_vs_instruct(
    data=ASR_BARS_DATA,
    output_path: str = "exp_runs/save_imgs/asr_bars.pdf",
    figsize=(3.45, 3.2),
    colors=None,
    base_alpha=0.32,
) -> Path:
    """Grouped bar chart: ASR base vs instruct, one panel per model family."""
    colors = colors or {"text": "#D55E00", "image": "#0072B2"}
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

    for ax, family in zip(axes, families):
        sizes = list(data[family].keys())
        x = np.arange(len(sizes))

        for channel in ("text", "image"):
            base_values = np.array([data[family][s][channel][0] for s in sizes], dtype=float)
            instruct_values = np.array([data[family][s][channel][1] for s in sizes], dtype=float)
            gains = instruct_values - base_values
            xpos = x + channel_offsets[channel]
            color = colors[channel]

            ax.bar(
                xpos,
                base_values,
                width=bar_w,
                facecolor=color,
                alpha=base_alpha,
                edgecolor=color,
                linewidth=1.0,
                zorder=2,
            )
            ax.bar(
                xpos,
                gains,
                width=bar_w,
                bottom=base_values,
                facecolor="white",
                edgecolor=color,
                linewidth=0.95,
                hatch="///",
                zorder=2,
            )

            for xpos_i, base_value, instruct_value, gain in zip(xpos, base_values, instruct_values, gains):
                ax.annotate(
                    f"{instruct_value:g}",
                    (xpos_i, instruct_value),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.0,
                    color=color,
                    fontweight="semibold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(sizes, fontsize=8.2)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}")
        ax.tick_params(axis="y", labelsize=7.6)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.grid(axis="y", linestyle="--", alpha=0.75)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

        ax.text(
            0.975,
            0.94,
            family,
            transform=ax.transAxes,
            ha="right",
            fontsize=8.5,
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="#C9CED6", linewidth=0.6),
            clip_on=False,
        )

    fig.supylabel("Attack Success Rate (%)", fontsize=7.9, x=0.075)

    input_handles = [
        Patch(facecolor=colors["text"], edgecolor=colors["text"], label="Text"),
        Patch(facecolor=colors["image"], edgecolor=colors["image"], label="Image"),
    ]
    component_handles = [
        Patch(facecolor="0.35", alpha=base_alpha, edgecolor="0.35", label="Base"),
        Patch(facecolor="white", edgecolor="0.35", hatch="///", label="Instruct - Base"),
    ]
    fig.legend(
        handles=input_handles,
        loc="upper left",
        bbox_to_anchor=(0.18, 0.995),
        ncol=2,
        frameon=False,
        fancybox=True,
        fontsize=7.6,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.65,
        borderpad=0.25,
    )
    fig.legend(
        handles=component_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.995),
        ncol=2,
        frameon=False,
        fancybox=True,
        fontsize=7.6,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.65,
        borderpad=0.25,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.03, 0, 1, 0.93], h_pad=0.45)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output


def agent_redteaming(
    output_path: str = "exp_runs/save_imgs/asr_text_vs_image.pdf",
    figsize: tuple[float, float] = (9.5, 6.6),
) -> Path:
    """Render ASR bars for text, image, and text→image transfer and save the figure."""
    data = {
        "InjectBench": {
            "gpt-5.4-mini": (98.2, 7.1, 0.0),
            # "gemini-3.1-flash": (100.0, 94.6, 80.4),
            "claude-haiku-4.5": (96.4, 10.7, 0.0),
            "kimi-k2.6": (85.7, 48.2, 50.0),
        },
        "AgentDojo": {
            "gpt-5.4-mini": (40.05, 4.0, 21.3),
            # "gemini-3.1-flash": (92, 30.7, 56.0),
            "claude-haiku-4.5": (1.3, 0.0, 0.0),
            "kimi-k2.6": (45.3, 20.0, 12.0),
        },
    }

    text_color = "#4C78A8"
    image_color = "#F58518"
    transfer_color = "#54A24B"
    label_fs = 16
    value_fs = 13

    benchmarks = list(data.keys())
    bar_width = 0.24
    group_offset = bar_width + 0.02
    tick_step = 30
    y_max = 106

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(len(benchmarks), 1, figsize=figsize, dpi=170, sharex=True, sharey=True)
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
            edgecolor="#35597A",
            linewidth=0.9,
        )
        bars_transfer = ax.bar(
            x,
            transfer_values,
            bar_width,
            label="Transfer (text→image)",
            color=transfer_color,
            edgecolor="#2D6A31",
            linewidth=0.9,
        )
        bars_image = ax.bar(
            x + group_offset,
            image_values,
            bar_width,
            label="Image",
            color=image_color,
            edgecolor="#B36312",
            linewidth=1.1,
        )

        max_value = float(max(text_values.max(), image_values.max(), transfer_values.max()))
        label_padding = max(1.0, max_value * 0.02)
        for series_idx, bars in enumerate((bars_text, bars_transfer, bars_image)):
            for bar in bars:
                value = bar.get_height()
                y_offset = label_padding + series_idx * label_padding * 0.35
                value_label = "0%" if np.isclose(value, 0.0) else f"{value:.1f}%"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + y_offset,
                    value_label,
                    ha="center",
                    va="bottom",
                    fontsize=value_fs,
                    clip_on=False,
                )

        benchmark_label = benchmark.replace("_", " ")
        ax.set_title(
            benchmark_label,
            fontsize=label_fs,
            loc="right",
            pad=8,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor="#F3F4F6",
                edgecolor="#9AA4B2",
                linewidth=1.1,
            ),
        )
        ax.set_ylim(0, y_max)
        ax.set_yticks(np.arange(0, y_max + 1, tick_step))
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=label_fs - 1)
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
        ax.tick_params(axis="y", labelsize=label_fs - 1)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_ax = axes[1] if len(axes) > 1 else axes[0]
    legend_ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=label_fs - 1,
    )
    fig.supylabel("Attack Success Rate (ASR %)", fontsize=label_fs, x=0.08)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.05, 0, 1, 1])
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


# ==================== Not used for now ============================

def plot_audio(output_path: str = "exp_runs/save_imgs/audio_headline_results.png") -> Path:
    models = ["gpt-audio-mini", "qwen3.5-omni-plus"]
    conds = ["Text", "Audio"]
    colors = {"gpt-audio-mini": "#4C78A8", "qwen3.5-omni-plus": "#F58518"}
    markers = {"Text": "o", "Audio": "s"}

    asr = np.array([[33.0, 1.5], [49.1, 13.4]])
    ut = np.array([[67.6, 86.9], [39.9, 81.8]])

    title_fs = 18
    label_fs = 18
    annot_fs = 18

    # Independent x-axis per panel; left panel zoomed to its data
    xlims = {"gpt-audio-mini": (60, 112), "qwen3.5-omni-plus": (30, 112)}
    xticks_per = {
        "gpt-audio-mini":    [60, 70, 80, 90, 100, 110],
        "qwen3.5-omni-plus": [40, 60, 80, 100],
    }

    # All boxes placed upper-right of their markers — they stay off the arrow path
    # since the arrow always travels down-right from Text to Audio.
    offsets = {
        ("gpt-audio-mini",   "Text"):  (18, 12),
        ("gpt-audio-mini",   "Audio"): (18, 12),
        ("qwen3.5-omni-plus", "Text"):  (18, 12),
        ("qwen3.5-omni-plus", "Audio"): (18, 12),
    }
    # Inline "switch to audio input" — centered below arrow midpoint of each panel
    arrow_label_off_pts = (0, -28)  # in offset points from arrow midpoint

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), dpi=170, sharey=True)

    for model_idx, model in enumerate(models):
        ax = axes[model_idx]
        x_text, x_audio = ut[model_idx, 0], ut[model_idx, 1]
        y_text, y_audio = asr[model_idx, 0], asr[model_idx, 1]

        # Arrow Text -> Audio. shrinkA/B leave a visible gap between arrow and marker.
        arrow = FancyArrowPatch(
            (x_text, y_text), (x_audio, y_audio),
            arrowstyle="-|>", mutation_scale=26,
            color=colors[model], lw=3.0, alpha=0.9, zorder=2,
            shrinkA=22, shrinkB=22,
        )
        ax.add_patch(arrow)

        # Inline label centered below the arrow midpoint
        mid_x, mid_y = (x_text + x_audio) / 2, (y_text + y_audio) / 2
        ax.annotate(
            "switch to\naudio input",
            xy=(mid_x-2, mid_y),
            xytext=arrow_label_off_pts,
            textcoords="offset points",
            fontsize=annot_fs - 2, ha="center", va="top",
            color=colors[model], style="italic", alpha=0.95,
        )

        # Points + boxed coordinate annotations
        for cond_idx, cond in enumerate(conds):
            x_val = ut[model_idx, cond_idx]
            y_val = asr[model_idx, cond_idx]
            ax.scatter(
                x_val, y_val,
                s=360, marker=markers[cond],
                color=colors[model], edgecolor="white",
                linewidth=2, zorder=5,
            )
            dx, dy = offsets[(model, cond)]
            ax.annotate(
                f"{cond}\n({x_val:.1f}, {y_val:.1f})",
                xy=(x_val, y_val), xytext=(dx, dy),
                textcoords="offset points",
                fontsize=annot_fs,
                ha="left" if dx > 0 else "right",
                va="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=colors[model], alpha=0.95, linewidth=1.6),
                zorder=4,
            )

        ax.axvline(100.0, color="#666666", linestyle="--", linewidth=1.4, alpha=0.7)

        ax.set_title(model, fontsize=title_fs, pad=14, fontweight="bold")
        ax.set_xlim(*xlims[model])
        ax.set_ylim(-6, 62)
        ax.set_xticks(xticks_per[model])
        ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=label_fs)
        ax.set_xlabel("Utility (UT %) under attack", fontsize=label_fs)

    axes[0].set_ylabel("Attack Success Rate (ASR %)", fontsize=label_fs)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
               markersize=13, label="Text input"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#555",
               markersize=13, label="Audio input"),
        Line2D([0], [0], color="#666666", linestyle="--", linewidth=1.8,
               label="No-attack utility (100%)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=True,
               fontsize=label_fs, bbox_to_anchor=(0.5, 1.06))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.subplots_adjust(wspace=0.10)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


    
if __name__ == "__main__":
    path = base_vs_instruct()
    print(f"Saved figure to {path}")
