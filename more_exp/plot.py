"""ASR comparison bar chart: text vs image / text vs audio across 4 models."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

ARROW_STYLE = '->'


def fig1():
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

    # (model, modality, text-ASR, defense-ASR)
    data = [
        ('gpt-5.4-mini',        'Image', 75.0,   0.0),
        ('claude-haiku-4.5',    'Image', 17.9,   0.0),
        ('gpt-audio-mini',      'Audio', 78.6,   8.9),
        ('qwen3.5-omni-plus',   'Audio', 100.0, 55.4),
    ]

    RED         = '#E2655F'
    RED_EDGE    = '#A32D2D'
    GREEN       = '#8AC247'
    GREEN_EDGE  = '#3B6D11'
    TEAL        = '#2DB7A0'
    TEAL_EDGE   = '#0F6E56'
    TITLE_BLUE  = '#185FA5'
    ARROW_YELLOW = '#E6A817'
    ARROW_BLUE   = '#185FA5'
    ARROW_LW     = 1.8
    TICK_GRAY   = '#5F5E5A'
    GRID_GRAY   = '#B4B2A9'

    fig, ax = plt.subplots(figsize=(9.0, 3))

    bar_w = 0.36
    pair_gap = 0.04
    group_gap = 0.55
    modality_gap = 0.95

    pos_t = []
    pos_d = []
    x = 0.0
    for i, (_, _, _, _) in enumerate(data):
        if i == 2:
            x += modality_gap
        elif i > 0:
            x += group_gap
        pos_t.append(x)
        pos_d.append(x + bar_w + pair_gap)
        x += bar_w * 2 + pair_gap

    for i, (name, mod, tv, dv) in enumerate(data):
        is_img = (mod == 'Image')
        d_fill = GREEN if is_img else TEAL
        d_edge = GREEN_EDGE if is_img else TEAL_EDGE

        ax.bar(pos_t[i], tv, bar_w, color=RED, edgecolor=RED_EDGE, linewidth=0.7, zorder=3)
        ax.bar(pos_d[i], dv, bar_w, color=d_fill, edgecolor=d_edge, linewidth=0.7, zorder=3)

        ax.text(pos_t[i], tv + 2.8, f'{tv:.1f}',
                ha='center', fontsize=10.5, fontweight='bold', color='#2C2C2A', zorder=5)
        label_d = f'{dv:.1f}' if dv > 0 else '0.0'
        ax.text(pos_d[i], max(dv, 0) + 2.8, label_d,
                ha='center', fontsize=10.5, fontweight='bold', color='#2C2C2A', zorder=5)

        if tv > 0:
            label_d_y = max(dv, 0) + 2.8
            x1, y1 = pos_t[i] + bar_w / 2, tv
            x2, y2 = pos_d[i], label_d_y + 5
            arr = FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle=ARROW_STYLE, mutation_scale=15,
                linestyle='-',
                linewidth=ARROW_LW,
                color=ARROW_YELLOW if is_img else ARROW_BLUE,
                zorder=4,
            )
            ax.add_patch(arr)

        pair_center = (pos_t[i] + pos_d[i]) / 2
        ax.text(pair_center, -6, name, ha='center', va='top',
                fontsize=10, color='#2C2C2A')

    img_pair_l = (pos_t[0] + pos_d[0]) / 2
    img_pair_r = (pos_t[1] + pos_d[1]) / 2
    aud_pair_l = (pos_t[2] + pos_d[2]) / 2
    aud_pair_r = (pos_t[3] + pos_d[3]) / 2
    img_center = (img_pair_l + img_pair_r) / 2
    aud_center = (aud_pair_l + aud_pair_r) / 2

    bracket_y = -16
    for left, right, label, color in [
        (pos_t[0] - bar_w*0.2, pos_d[1] + bar_w*0.2, 'Image defense',  GREEN_EDGE),
        (pos_t[2] - bar_w*0.2, pos_d[3] + bar_w*0.2, 'Audio defense',  TEAL_EDGE),
    ]:
        ax.plot([left, right], [bracket_y, bracket_y], color=color, linewidth=1.1, clip_on=False)
        ax.plot([left, left], [bracket_y, bracket_y + 1.8], color=color, linewidth=1.1, clip_on=False)
        ax.plot([right, right], [bracket_y, bracket_y + 1.8], color=color, linewidth=1.1, clip_on=False)
        ax.text((left + right) / 2, bracket_y - 4, label, ha='center', va='top',
                fontsize=10, fontweight='bold', color=color)

    ax.set_ylim(-26, 118)
    ax.set_xlim(pos_t[0] - 0.45, pos_d[-1] + 0.45)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=10.5, color='#2C2C2A')
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis='y', labelsize=9.5, colors=TICK_GRAY, length=3)
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_color(GRID_GRAY)
    ax.spines['left'].set_linewidth(0.6)
    ax.grid(axis='y', linestyle=(0, (1.5, 3)), linewidth=0.5, color=GRID_GRAY, alpha=0.6, zorder=1)
    ax.set_axisbelow(True)


    arrow_ls = '-'
    legend_elements = [
        Line2D([0, 1], [0, 0], color=ARROW_YELLOW, lw=ARROW_LW,
               linestyle=arrow_ls, marker='>', markevery=[1],
               markersize=9, markerfacecolor=ARROW_YELLOW,
               markeredgecolor=ARROW_YELLOW, label='Text to Image'),
        Line2D([0, 1], [0, 0], color=ARROW_BLUE, lw=ARROW_LW,
               linestyle=arrow_ls, marker='>', markevery=[1],
               markersize=9, markerfacecolor=ARROW_BLUE,
               markeredgecolor=ARROW_BLUE, label='Text to Audio'),
    ]
    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, 1.02), ncol=2,
              frameon=False, fontsize=10,
              handlelength=3.0, handletextpad=0.7, columnspacing=3.0)

    plt.tight_layout()
    plt.savefig('exp_runs/asr_chart.pdf', bbox_inches='tight', pad_inches=0.12)
    plt.savefig('exp_runs/asr_chart.png', bbox_inches='tight', pad_inches=0.12)

    print('Saved asr_chart.pdf!!!')


if __name__ == '__main__':
    fig1()