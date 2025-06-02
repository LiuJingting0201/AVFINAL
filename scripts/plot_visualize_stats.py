import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')


nice_colors = ["#a6badf", "#4675b0", "#28437d"]  
labels = ['LiDAR', 'Radar', 'Fusion']
fontsize = 15
label_fontsize = 12
legend_fontsize = 11
bar_width = 0.22

plot_dir = r"D:\MyProjects\AV4\output\plots"# Change this to your desired output directory
os.makedirs(plot_dir, exist_ok=True)

def plot_grouped_bar(df, xlabels, title, xlabel, filename):
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(max(7,len(df)*0.65), 4.4))
    vertical_offsets = [0.012, 0.045, 0.078]  # Layered offset (going upward)
    for i, col in enumerate(['lidar_recall','radar_recall','fusion_recall']):
        ax.bar(
            x + (i-1)*bar_width, df[col], width=bar_width,
            color=nice_colors[i], label=labels[i],
            edgecolor='none', alpha=0.93
        )
        # Show numbers with layered offset
        for xi, val in zip(x + (i-1)*bar_width, df[col]):
            ax.text(
                xi, val + vertical_offsets[i], f"{val:.2f}",
                ha='center', va='bottom', fontsize=10, color='#303033', weight='semibold'
            )
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=27 if len(xlabels)>5 else 0, ha='right', fontsize=label_fontsize)
    ax.set_ylim(0, 1.09)
    ax.set_ylabel('Recall', fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize+1, pad=8)
    ax.legend(fontsize=legend_fontsize, frameon=False, loc='lower left')
    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), dpi=380, bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir, filename.replace('.png', '.svg')), bbox_inches='tight')
    plt.close()

# Plotting recall statistics from CSV files，change paths as needed
# By scene
df_scene = pd.read_csv(r"D:\MyProjects\AVFinal\output\recall_by_scene.csv", index_col=0)
plot_grouped_bar(df_scene, df_scene.index.tolist(), "Recall by Scene", "Scene", "recall_by_scene.png")

# By category
df_cat = pd.read_csv(r"D:\MyProjects\AVFinal\output\recall_by_category.csv", index_col=0)
plot_grouped_bar(df_cat, df_cat.index.tolist(), "Recall by Category", "Category", "recall_by_category.png")

# By distance
df_dist = pd.read_csv(r"D:\MyProjects\AVFinal\output\recall_by_distance.csv", index_col=0)
plot_grouped_bar(df_dist, df_dist.index.tolist(), "Recall by Distance Bin", "Distance Bin", "recall_by_distance.png")

print("plots saved to", plot_dir)
