import matplotlib.pyplot as plt
import numpy as np




def plot_fig1():
    SCORE_FS  = 10
    LABEL_FS  = 12
    TICK_FS   = 11
    LEGEND_FS = 12

    color_baseline = '#7DD4E8'
    color_ours     = '#1A7AAF'
    width = 0.34

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.9))

    # ---- Subfigure 1: DP-SGD epsilon ----
    data1 = {
        'CIFAR-10':  (1.2867, 1.7832),
        'CIFAR-100': (1.2427, 1.8837),
        'CINIC-10':  (0.7917, 1.3849),
    }

    labels1 = list(data1.keys())
    baseline_vals1 = [v[0] for v in data1.values()]
    ours_vals1     = [v[1] for v in data1.values()]
    x1 = np.arange(len(labels1))

    bars1 = ax1.bar(x1 - width/2, baseline_vals1, width, label='Steinke et al. 2023', color=color_baseline, edgecolor='none')
    bars2 = ax1.bar(x1 + width/2, ours_vals1,     width, label='Ours',                color=color_ours,     edgecolor='none')

    for bar in list(bars1) + list(bars2):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=SCORE_FS, color='#333333')

    ax1.set_ylabel('Empirical Epsilon', fontsize=LABEL_FS)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(labels1, fontsize=TICK_FS)
    ax1.tick_params(axis='y', labelsize=TICK_FS)
    ax1.set_ylim(0, 2.2)
    ax1.yaxis.set_ticks(np.arange(0, 2.5, 0.5))
    ax1.yaxis.grid(False)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # ax1.set_title('DP-SGD', fontsize=18, fontweight='bold', pad=55)

    # ---- Subfigure 2: CIFAR-10 TPR ----
    data2 = {
        'Undefended': (23.3813, 84.9813),
        'RelaxLoss':  (1.0563, 21.4500),
        'DP-SGD':     (0.6500, 2.5063),

    }

    labels2 = list(data2.keys())
    baseline_vals2 = [v[0] for v in data2.values()]
    ours_vals2     = [v[1] for v in data2.values()]
    x2 = np.arange(len(labels2))

    bars3 = ax2.bar(x2 - width/2, baseline_vals2, width, label='Steinke et al. 2023', color=color_baseline, edgecolor='none')
    bars4 = ax2.bar(x2 + width/2, ours_vals2,     width, label='Ours',                color=color_ours,     edgecolor='none')

    for bar in list(bars3) + list(bars4):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=SCORE_FS, color='#333333')

    ax2.set_ylabel('TPR @ 0.1% FPR (%)', fontsize=LABEL_FS)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2, fontsize=TICK_FS)
    ax2.tick_params(axis='y', labelsize=TICK_FS)
    ax2.set_ylim(0, 100)
    ax2.yaxis.grid(False)
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    # ax2.set_title('CIFAR-10', fontsize=18, fontweight='bold', pad=55)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=LEGEND_FS, loc='upper center',
               bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False,
               handlelength=1.4, columnspacing=1.6)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('combined_barplot.pdf', dpi=300, bbox_inches='tight')
    # plt.savefig('combined_barplot.png', dpi=300, bbox_inches='tight')

def neighbors():
    def norm_pdf(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    RED    = "#E24B4A"   # OUT / non-member
    BLUE   = "#378ADD"   # IN / member
    GREEN  = "#1D9E75"   # Ideal neighbor (approximates OUT)
    PURPLE = "#9B59B6"   # Too-close neighbor (approximates IN)

    # Ground-truth distributions: solid + filled (the references).
    # Neighbor distributions: dashed, with their own distinct colors.
    #   Ideal neighbor   (mu=13.50) tracks OUT
    #   Too-close neighbor (mu=10.25) tracks IN
    # (name, mu, sigma, color, linestyle, linewidth, filled)
    distributions = [
        ("OUT (non-member)",   13.73, 3.06, RED,    "-",  3.0, True),
        ("IN (member)",         9.28, 2.40, BLUE,   "-",  3.0, True),
        ('"Good" Neighbors',    13.50, 3.49, GREEN,  "--", 2.6, False),
        ('"Too-close" Neighbors',10.25, 2.52, PURPLE, "--", 2.6, False),
    ]

    # Sizing tuned for a single column in a two-column paper.
    LEGEND_FS = 8
    ANNOT_FS  = 8

    x = np.linspace(0, 25, 1000)

    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    handles = {}
    for name, mu, sigma, color, ls, lw, filled in distributions:
        y = norm_pdf(x, mu, sigma)
        h, = ax.plot(x, y, label=name, color=color, linestyle=ls, linewidth=lw)
        if filled:
            ax.fill_between(x, y, alpha=0.14, color=color, linewidth=0)
        handles[name] = h

    # Compact, color-matched tags carry the takeaway:
    #   Ideal neighbor  -> looks like OUT;  Too-close neighbor -> looks like IN.
    ax.annotate(r"$\approx$ OUT",
                xy=(16.0, norm_pdf(16.0, 13.50, 3.49)), xycoords="data",
                xytext=(20.5, 0.072), textcoords="data",
                fontsize=ANNOT_FS, color=RED, ha="center", va="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1,
                                connectionstyle="arc3,rad=-0.2"))
    ax.annotate(r"$\approx$ IN",
                xy=(10.25, norm_pdf(10.25, 10.25, 2.52)), xycoords="data",
                xytext=(4.0, 0.135), textcoords="data",
                fontsize=ANNOT_FS, color=BLUE, ha="center", va="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1,
                                connectionstyle="arc3,rad=0.2"))

    # Two separate legends: ground-truth pinned to the left, neighbors to the right.
    leg_left = ax.legend(
        handles=[handles["OUT (non-member)"], handles["IN (member)"]],
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.30),
        fontsize=LEGEND_FS,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.4,
    )
    ax.add_artist(leg_left)
    ax.legend(
        handles=[handles['"Good" Neighbors'], handles['"Too-close" Neighbors']],
        loc="upper right",
        bbox_to_anchor=(1.02, 1.30),
        fontsize=LEGEND_FS,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.4,
    )

    ax.set_xlim(0, 25)
    ax.set_ylim(0, None)
    ax.set_xticks([])
    ax.yaxis.set_visible(False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0))

    plt.tight_layout()
    plt.savefig("lira_distributions.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("lira_distributions.png", dpi=200, bbox_inches="tight")


if __name__ == '__main__':
    plot_fig1()
    neighbors()
