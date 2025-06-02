import os
import json
import pandas as pd
import numpy as np

def points_in_bbox(df, box):
    cx, cy, cz = box['translation']
    dx, dy, dz = box['size']
    # Simple AABB bounding box check without considering rotation
    x_min, x_max = cx - dx/2, cx + dx/2
    y_min, y_max = cy - dy/2, cy + dy/2
    z_min, z_max = cz - dz/2, cz + dz/2
    pts = df[
        (df['X'] >= x_min) & (df['X'] <= x_max) &
        (df['Y'] >= y_min) & (df['Y'] <= y_max) &
        (df['Z'] >= z_min) & (df['Z'] <= z_max)
    ]
    return len(pts)

def analyze_one_sample(data_dir, out_csv='fusion_stats.csv'):
    ann_path = os.path.join(data_dir, 'annotations.json')
    lidar_csv = os.path.join(data_dir, 'merged_lidar.csv')
    radar_csv = os.path.join(data_dir, 'merged_radar.csv')

    # Check if required files exist
    if not (os.path.exists(ann_path) and os.path.exists(lidar_csv) and os.path.exists(radar_csv)):
        print(f"Missing files in {data_dir}, skipping...")
        return

    try:
        with open(ann_path, 'r') as f:
            annotations = json.load(f)
        lidar_df = pd.read_csv(lidar_csv)
        radar_df = pd.read_csv(radar_csv)
        fusion_df = pd.concat([lidar_df, radar_df], ignore_index=True)
    except Exception as e:
        print(f"Error reading data in {data_dir}: {e}")
        return

    stats = []
    for box in annotations:
        cat = box.get('category_name', 'unknown')
        box_id = box.get('token', 'unknown')
        cx, cy, cz = box['translation']
        dist = np.linalg.norm([cx, cy, cz])
        dx, dy, dz = box['size']
        volume = dx * dy * dz
        n_lidar = points_in_bbox(lidar_df, box)
        n_radar = points_in_bbox(radar_df, box)
        n_fusion = points_in_bbox(fusion_df, box)
        stat = {
            'box_id': box_id,
            'category': cat,
            'distance': dist,
            'volume': volume,
            'lidar_points': n_lidar,
            'radar_points': n_radar,
            'fusion_points': n_fusion,
            'lidar_density': n_lidar / volume if volume > 0 else 0,
            'radar_density': n_radar / volume if volume > 0 else 0,
            'fusion_density': n_fusion / volume if volume > 0 else 0,
        }
        stats.append(stat)
    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(os.path.join(data_dir, out_csv), index=False)
    print(f"Sample {data_dir} finished. Results saved to {out_csv}.")
    # Calculate recall (number of detected bounding boxes)
    recall_lidar = (df_stats['lidar_points'] > 0).sum() / len(df_stats)
    recall_radar = (df_stats['radar_points'] > 0).sum() / len(df_stats)
    recall_fusion = (df_stats['fusion_points'] > 0).sum() / len(df_stats)
    print(f"Recall: LiDAR={recall_lidar:.2%}, Radar={recall_radar:.2%}, Fusion={recall_fusion:.2%}")
    return df_stats

if __name__ == "__main__":
    print("Script started!")
    parent_dir = r"D:\MyProjects\AV4\output"
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

    for scene in scenes:
        scene_path = os.path.join(parent_dir, scene)
        if not os.path.isdir(scene_path): continue
        for sample in os.listdir(scene_path):
            data_dir = os.path.join(scene_path, sample, 'data')
            if os.path.isdir(data_dir):
                analyze_one_sample(data_dir)
