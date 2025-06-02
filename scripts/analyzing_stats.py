import os
import pandas as pd

parent_dir = r"D:\MyProjects\AVFinal\output"
scenes = [
    "scene_bridge_city",
    "scene_clear_city",
    "scene_construction",
    "scene_night_highway",
    "scene_overcast_city",
    "scene_rainy_highway",
    "scene_snow_city",
    "scene_terminal_area",
    "scene_twilight"
]

all_stats = []
for scene in scenes:
    scene_path = os.path.join(parent_dir, scene)
    if not os.path.isdir(scene_path):
        continue
    for sample in os.listdir(scene_path):
        data_dir = os.path.join(scene_path, sample, 'data')
        stats_path = os.path.join(data_dir, 'fusion_stats.csv')
        if os.path.exists(stats_path):
            df = pd.read_csv(stats_path)
            df['scene'] = scene  # Scene label
            all_stats.append(df)

df_all = pd.concat(all_stats, ignore_index=True)
df_all.to_csv(os.path.join(parent_dir, 'all_fusion_stats.csv'), index=False)

# Scene-level recall
recall_by_scene = df_all.groupby('scene').apply(
    lambda x: pd.Series({
        'lidar_recall': (x['lidar_points'] > 0).mean(),
        'radar_recall': (x['radar_points'] > 0).mean(),
        'fusion_recall': (x['fusion_points'] > 0).mean(),
        'num_boxes': len(x)
    })
)
recall_by_scene.to_csv(os.path.join(parent_dir, 'recall_by_scene.csv'))
print("\n=== Scene-level Recall ===")
print(recall_by_scene)

# Category recall
recall_by_category = df_all.groupby('category').apply(
    lambda x: pd.Series({
        'lidar_recall': (x['lidar_points'] > 0).mean(),
        'radar_recall': (x['radar_points'] > 0).mean(),
        'fusion_recall': (x['fusion_points'] > 0).mean(),
        'num_boxes': len(x)
    })
)
recall_by_category.to_csv(os.path.join(parent_dir, 'recall_by_category.csv'))
print("\n=== Category Recall ===")
print(recall_by_category)


# ======= Distance-based recall statistics =======

# Define distance bins
bins = [0, 20, 40, 60, 200]
labels = ['0-20m', '20-40m', '40-60m', '60m+']
df_all['distance_bin'] = pd.cut(df_all['distance'], bins=bins, labels=labels, right=False)

# Distance-based recall
recall_by_distance = df_all.groupby('distance_bin').apply(
    lambda x: pd.Series({
        'lidar_recall': (x['lidar_points'] > 0).mean(),
        'radar_recall': (x['radar_points'] > 0).mean(),
        'fusion_recall': (x['fusion_points'] > 0).mean(),
        'num_boxes': len(x)
    })
)
recall_by_distance.to_csv(os.path.join(parent_dir, 'recall_by_distance.csv'))
print("\n=== Distance-based Recall ===")
print(recall_by_distance)