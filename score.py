import matplotlib.pyplot as plt
import numpy as np

scores = [
    {
        "Data Story": {
            "Insightfulness": 8.5,
            "Comprehensiveness": 9.5,
            "Effectiveness": 9,
        },
        "Data Visualization": {
            "Understandability": 9,
            "Aestheticness": 9,
            "Expressiveness": 9.5,
            "Interactivity": 8,
        },
        "Visualization Style": {
            "Aestheticness": 8.5,
            "Interactivity": 7.5,
            "Integrity": 9,
        },
        "Infographic-style INV": {
            "Engagement": 8,
            "Usefulness": 9,
            "Legibility": 9,
            "Design": 8,
            "Aestheticness": 9,
        },
    },
    {
        "Data Story": {
            "Insightfulness": 9,
            "Comprehensiveness": 8.5,
            "Effectiveness": 8,
        },
        "Data Visualization": {
            "Understandability": 9.5,
            "Aestheticness": 9,
            "Expressiveness": 9,
            "Interactivity": 8.5,
        },
        "Visualization Style": {"Aestheticness": 9, "Interactivity": 8, "Integrity": 9},
        "Infographic-style INV": {
            "Engagement": 8,
            "Usefulness": 9,
            "Legibility": 9,
            "Design": 8.5,
            "Aestheticness": 9.5,
        },
    },
    {
        "Data Story": {
            "Insightfulness": 9,
            "Comprehensiveness": 9.5,
            "Effectiveness": 9,
        },
        "Data Visualization": {
            "Understandability": 8.5,
            "Aestheticness": 8.5,
            "Expressiveness": 8.5,
            "Interactivity": 8,
        },
        "Visualization Style": {
            "Aestheticness": 9,
            "Interactivity": 8,
            "Integrity": 9,
        },
        "Infographic-style INV": {
            "Engagement": 8,
            "Usefulness": 10,
            "Legibility": 9,
            "Design": 8,
            "Aestheticness": 9,
        },
    },
    {
        "Data Story": {
            "Insightfulness": 9,
            "Comprehensiveness": 9,
            "Effectiveness": 9,
        },
        "Data Visualization": {
            "Understandability": 9,
            "Aestheticness": 9,
            "Expressiveness": 9.5,
            "Interactivity": 7,
        },
        "Visualization Style": {
            "Aestheticness": 9,
            "Interactivity": 8.5,
            "Integrity": 9.5,
        },
        "Infographic-style INV": {
            "Engagement": 8,
            "Usefulness": 9,
            "Legibility": 9,
            "Design": 8,
            "Aestheticness": 9,
        },
    },
    {
        "Data Story": {
            "Insightfulness": 8,
            "Comprehensiveness": 8.5,
            "Effectiveness": 9,
        },
        "Data Visualization": {
            "Understandability": 8.5,
            "Aestheticness": 9,
            "Expressiveness": 8.5,
            "Interactivity": 7.5,
        },
        "Visualization Style": {
            "Aestheticness": 9.5,
            "Interactivity": 7,
            "Integrity": 9,
        },
        "Infographic-style INV": {
            "Engagement": 7,
            "Usefulness": 9,
            "Legibility": 9,
            "Design": 8,
            "Aestheticness": 9,
        },
    },
    {
        "Data Story": {
            "Insightfulness": 8.5,
            "Comprehensiveness": 8.5,
            "Effectiveness": 9,
        },
        "Data Visualization": {
            "Understandability": 8,
            "Aestheticness": 9,
            "Expressiveness": 9,
            "Interactivity": 7.5,
        },
        "Visualization Style": {
            "Aestheticness": 8.5,
            "Interactivity": 7.5,
            "Integrity": 9,
        },
        "Infographic-style INV": {
            "Engagement": 7,
            "Usefulness": 9,
            "Legibility": 9,
            "Design": 9,
            "Aestheticness": 9,
        },
    },
    {
        "Data Story": {
            "Insightfulness": 9.5,
            "Comprehensiveness": 10,
            "Effectiveness": 9,
        },
        "Data Visualization": {
            "Understandability": 9,
            "Aestheticness": 8.5,
            "Expressiveness": 9,
            "Interactivity": 7.5,
        },
        "Visualization Style": {
            "Aestheticness": 9,
            "Interactivity": 7.5,
            "Integrity": 9,
        },
        "Infographic-style INV": {
            "Engagement": 7.5,
            "Usefulness": 9,
            "Legibility": 9,
            "Design": 8,
            "Aestheticness": 9,
        },
    },
    {
        "Data Story": {
            "Insightfulness": 8.5,
            "Comprehensiveness": 8.5,
            "Effectiveness": 8,
        },
        "Data Visualization": {
            "Understandability": 9,
            "Aestheticness": 9,
            "Expressiveness": 9,
            "Interactivity": 8,
        },
        "Visualization Style": {
            "Aestheticness": 9.5,
            "Interactivity": 7.5,
            "Integrity": 9.5,
        },
        "Infographic-style INV": {
            "Engagement": 7.5,
            "Usefulness": 8,
            "Legibility": 9,
            "Design": 9,
            "Aestheticness": 8,
        },
    },
    {
        "Data Story": {
            "Insightfulness": 8.5,
            "Comprehensiveness": 9,
            "Effectiveness": 8.5,
        },
        "Data Visualization": {
            "Understandability": 9,
            "Aestheticness": 9,
            "Expressiveness": 9,
            "Interactivity": 7,
        },
        "Visualization Style": {
            "Aestheticness": 9,
            "Interactivity": 7.5,
            "Integrity": 9,
        },
        "Infographic-style INV": {
            "Engagement": 7,
            "Usefulness": 9,
            "Legibility": 8,
            "Design": 9,
            "Aestheticness": 8,
        },
    },
    {
        "Data Story": {
            "Insightfulness": 9,
            "Comprehensiveness": 9,
            "Effectiveness": 8.5,
        },
        "Data Visualization": {
            "Understandability": 8.5,
            "Aestheticness": 8.5,
            "Expressiveness": 9,
            "Interactivity": 7.5,
        },
        "Visualization Style": {"Aestheticness": 9, "Interactivity": 8, "Integrity": 9},
        "Infographic-style INV": {
            "Engagement": 7.5,
            "Usefulness": 9,
            "Legibility": 8,
            "Design": 7.5,
            "Aestheticness": 9,
        },
    },
]


def calc_avg():
    # 计算scores中每个维度的平均值
    # 初始化存储所有评分的字典
    all_scores = {}

    # 遍历所有评分数据
    for score_dict in scores:
        for category, dimensions in score_dict.items():
            if category not in all_scores:
                all_scores[category] = {}
            for dimension, score in dimensions.items():
                if dimension not in all_scores[category]:
                    all_scores[category][dimension] = []
                all_scores[category][dimension].append(score)

    # 计算平均值
    averages = {}
    for category, dimensions in all_scores.items():
        averages[category] = {}
        for dimension, scores_list in dimensions.items():
            averages[category][dimension] = sum(scores_list) / len(scores_list)

    # 打印平均值
    for category, dimensions in averages.items():
        print(f"{category}:")
        for dimension, avg in dimensions.items():
            print(f"  {dimension}: {avg:.2f}")
        print()


def draw_image():
    # Data
    groups = [
        (
            "Data Story",
            [
                ("Insightfulness", 8.75),
                ("Comprehensiveness", 9.00),
                ("Effectiveness", 8.70),
            ],
        ),
        (
            "Data Visualization",
            [
                ("Understandability", 8.80),
                ("Aestheticness", 8.85),
                ("Expressiveness", 9.00),
                ("Interactivity", 7.65),
            ],
        ),
        (
            "Visualization Style",
            [
                ("Aestheticness", 9.00),
                ("Interactivity", 7.70),
                ("Integrity", 9.10),
            ],
        ),
        (
            "Infographic-style INV",
            [
                ("Engagement", 7.55),
                ("Usefulness", 9.00),
                ("Legibility", 8.80),
                ("Design", 8.30),
                ("Aestheticness", 8.85),
            ],
        ),
    ]

    # Colors for each top-level group
    group_colors = {
        "Data Story": "#F15BB5",  # pink
        "Data Visualization": "#2EC4B6",  # teal
        "Visualization Style": "#FF9F1C",  # orange
        "Infographic-style INV": "#9B5DE5",  # purple
    }

    # Flatten data with group separators
    labels = []
    values = []
    colors = []
    group_positions = []
    x = 0
    spacer = 0.8  # visual spacer between groups
    tick_positions = []
    tick_labels = []

    for group_name, items in groups:
        group_positions.append(x)
        for label, val in items:
            labels.append(f"{label}")
            values.append(val)
            colors.append(group_colors[group_name])
            tick_positions.append(x)
            tick_labels.append(label)
            x += 1
        # add spacer
        x += spacer

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
    bars = ax.bar(
        tick_positions,
        values,
        color=colors,
        width=0.8,
        edgecolor="white",
        linewidth=0.5,
    )

    # Value labels
    for rect, val in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.2,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )

    # Axes formatting
    ax.set_ylim(0, 10.5)
    ax.set_ylabel("Rating (0–10)", fontsize=11)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=9)

    # Gridlines
    ax.yaxis.grid(True, color="#E5E7EB", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_axisbelow(True)

    # Group headers with brackets or background bands
    for group_name, items in groups:
        n = len(items)
        start = group_positions[groups.index((group_name, items))]
        end = start + n - 1
        mid = (start + end) / 2
        # group label
        ax.text(
            mid,
            10.9,
            group_name,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
        # subtle separator line at group end (after last bar)
        ax.axvline(x=end + 0.45, color="#F3F4F6", linewidth=12, alpha=0.8)

    # Legend (manual handles)
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=color, edgecolor="white", label=name)
        for name, color in group_colors.items()
    ]
    ax.legend(
        handles=legend_handles,
        title="Category",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.02),
    )

    # # Title
    # ax.set_title("Ratings by category and dimension", fontsize=14, pad=12)

    plt.tight_layout()
    plt.savefig("ratings_chart.svg", format="svg")
    print("Saved to ratings_chart.svg")
    plt.show()


if __name__ == "__main__":
    # calc_avg()
    draw_image()
