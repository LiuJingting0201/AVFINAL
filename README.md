# TforAV_Final
AVFINAL is a modular toolkit for multi-sensor fusion, visualization, and statistical analysis on the MAN TruckScenes dataset.
ref：https://github.com/TUMFTM/truckscenes-devkit   https://brandportal.man/d/QSf8mPdU5Hgj
It enables efficient LiDAR & Radar fusion, batch scenario analysis, and visual/statistical reporting for advanced autonomous driving research.


Environment Setup
1. Python Environment
Python 3.8+ recommended

Use a virtual environment if possible (venv or conda)

2. Dependencies

Key packages:

numpy, matplotlib, open3d, pandas, scipy, plotly

Official TruckScenes devkit:

```pip install truckscenes-devkit[all]```
3. Dataset Preparation
Download the MAN TruckScenes dataset (academic request required)

data/man-truckscenes/v1.0-mini/
Make sure meta files like scene.json, sample.json are present in v1.0-mini.
For large datasets, you can link the data directory with a Windows junction:

Directory Structure
AVFINAL/
├── configs/                      # Scenario/sample configs
│   └── selected_samples.json
├── data/                         # Dataset root
│   └── man-truckscenes/
|       ├── samples
|       ├── sweeps
│       └── v1.0-mini/
├── output/                       # Outputs (figures, stats)
│   ├── scene_xxx/
│   └── summary/
├── reports/                      # Project reports, docs (optional)
├── scripts/                      # Core Python scripts
│   ├── sample_selector.py
│   ├── output_management.py
│   ├── fusion_stats.py
│   ├── analyzing_stats.py
│   ├── visualize_fused_pts_and_3d_boxes.py
│   └── plot_visualize_stats.py
└── README.md

Core Script Descriptions
sample_selector.py
Scenario-based sample selection. Generates selected_samples.json for batch processing.

visualize_fused_pts_and_3d_boxes.py
Visualizes sensor fusion for a single sample (LiDAR, Radar, 3D boxes). Useful for debugging or figure generation.

output_management.py
Batch pipeline controller. Processes all selected samples for fusion, visualization, and data export.

fusion_stats.py
Calculates per-sample and per-object metrics (recall, density) and saves CSV files.

analyzing_stats.py
Aggregates all statistics, computes global/scene/category/distance-level results for reporting.



Usage: Step-by-Step
1. Select Typical Samples by Scenario
```python scripts/sample_selector.py```
This creates or updates configs/selected_samples.json listing all the scene/sample tokens for later steps.

2. Batch Fusion, Visualization, and Export
```python scripts/output_management.py```
Runs the full pipeline: point cloud fusion, box alignment, and image/CSV/HTML export for every selected sample. visualize_fused_pts_and_3d_boxes was imported.

3. Aggregate Global Statistics
```python scripts/analyzing_stats.py```
Produces summary CSVs (e.g., recall by scene/category/distance) under output/summary/.

4. plot visiable Stat.
```python plot_visualize_stats.py```

5. (Optional) Visualize a Single Sample
python scripts/visualize_fused_pts_and_3d_boxes.py
Customize sample_token in the script for your case.

Results & Outputs
All results (figures, point clouds, statistics) are saved under the output/ directory.

Visualizations are organized by scenario, with summary CSVs in output/summary/.

HTML/PNG outputs and statistics are available per sample/scene.


Tips & Notes
To process different scenarios, edit configs/selected_samples.json or update logic in sample_selector.py.
To change visualization settings (viewpoint, distance, coloring), edit visualize_fused_pts_and_3d_boxes.py.
For any issues, check your data paths, sample tokens, and all required dependencies.

