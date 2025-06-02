#!/usr/bin/env python3
"""
TruckScenes Output Management Module
Features:
1. Batch process selected samples
2. Save visualization images, point cloud data, and statistics
3. Generate summary reports and CSV files
4. Error handling and logging
New Features:
1. Multi-view saving: side view, front view, bird's-eye view
2. Support for multiple 3D formats: PLY, OBJ, HTML interactive 3D
3. Improved VSCode compatibility
4. Save camera parameters for future use
"""

import os
import json
import traceback
import pandas as pd
import numpy as np
import inspect
from datetime import datetime
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Attempt to import open3d (used for saving point clouds)
try:
    import open3d as o3d
    O3D_AVAILABLE = True
    print("✓ Open3D available for point cloud saving")
except ImportError:
    O3D_AVAILABLE = False
    print("⚠️ Open3D not available - point clouds will not be saved")

# Import main visualization class
try:
    from visualize_fused_pts_and_3d_boxes import TruckScenesVisualizer
    print("✓ TruckScenesVisualizer imported successfully")
except ImportError as e:
    print(f"❌ Failed to import TruckScenesVisualizer: {e}")
    print("Please ensure visualize_fused_pts_and_3d_boxes.py is in the same directory")


# ==== 1. Paths and Configuration ====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "man-truckscenes")
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")
OUTPUT_ROOT = os.path.join(BASE_DIR, "output")
SELECTED_JSON = os.path.join(CONFIGS_DIR, "selected_samples.json")
SUMMARY_DIR = os.path.join(OUTPUT_ROOT, "summary")

# Ensure directories exist
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)

print(f"📁 Configuration Paths:")
print(f"  - Data Directory: {DATA_DIR}")
print(f"  - Config Directory: {CONFIGS_DIR}")
print(f"  - Output Directory: {OUTPUT_ROOT}")
print(f"  - Selected Samples: {SELECTED_JSON}")


# ==== 2. Enhanced Save Functions ====
def ensure_dir(path):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)


def save_point_cloud_multi_format(points, base_filename, colors=None):
    """
    Save point cloud in multiple formats
    
    Args:
        points: numpy array (N, 3)
        base_filename: Base filename (without extension)
        colors: numpy array (N, 3) Optional color information
    
    Returns:
        dict: Dictionary of saved file paths
    """
    if points is None or len(points) == 0:
        print(f"⚠️ Warning: No points to save for {base_filename}")
        return {}
    
    saved_files = {}
    
    # 1. Save as simple TXT format (viewable directly in VSCode)
    txt_path = f"{base_filename}.txt"
    try:
        if colors is not None:
            data_to_save = np.hstack([points, colors])
            header = "# X Y Z R G B"
        else:
            data_to_save = points
            header = "# X Y Z"
        
        np.savetxt(txt_path, data_to_save, 
                  fmt='%.6f', delimiter=' ', header=header)
        saved_files['txt'] = txt_path
        print(f"✓ Saved TXT point cloud: {os.path.basename(txt_path)} ({len(points):,} points)")
    except Exception as e:
        print(f"❌ Error saving TXT {txt_path}: {e}")
    
    # 2. Save as PLY format (supports colors)
    if O3D_AVAILABLE:
        ply_path = f"{base_filename}.ply"
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(ply_path, pcd)
            saved_files['ply'] = ply_path
            print(f"✓ Saved PLY point cloud: {os.path.basename(ply_path)}")
        except Exception as e:
            print(f"❌ Error saving PLY {ply_path}: {e}")
    
    # 3. Save as simple CSV format (Excel and other tools friendly)
    csv_path = f"{base_filename}.csv"
    try:
        if colors is not None:
            df = pd.DataFrame(np.hstack([points, colors]), 
                            columns=['X', 'Y', 'Z', 'R', 'G', 'B'])
        else:
            df = pd.DataFrame(points, columns=['X', 'Y', 'Z'])
        df.to_csv(csv_path, index=False)
        saved_files['csv'] = csv_path
        print(f"✓ Saved CSV point cloud: {os.path.basename(csv_path)}")
    except Exception as e:
        print(f"❌ Error saving CSV {csv_path}: {e}")
    
    return saved_files


def create_interactive_3d_html(lidar_points, radar_points, annotations, 
                               sample_token, save_path):
    """
    Create interactive 3D HTML visualization
    
    Args:
        lidar_points: LiDAR point cloud
        radar_points: Radar point cloud  
        annotations: Bounding boxes
        sample_token: Sample ID
        save_path: Save path
    """
    try:
        fig = go.Figure()
        
        # Add LiDAR point cloud
        if len(lidar_points) > 0:
            # Downsample to improve performance
            step = max(1, len(lidar_points) // 20000)
            viz_lidar = lidar_points[::step]
            
            # Compute distances for coloring
            distances = np.linalg.norm(viz_lidar[:, :2], axis=1)
            
            fig.add_trace(go.Scatter3d(
                x=viz_lidar[:, 0],
                y=viz_lidar[:, 1], 
                z=viz_lidar[:, 2],
                mode='markers',
                marker=dict(
                    size=1,
                    color=distances,
                    colorscale='Viridis',
                    colorbar=dict(title="Distance (m)"),
                    opacity=0.6
                ),
                name=f'LiDAR ({len(viz_lidar):,} points)',
                hovertemplate='<b>LiDAR Point</b><br>' +
                             'X: %{x:.2f}m<br>' +
                             'Y: %{y:.2f}m<br>' + 
                             'Z: %{z:.2f}m<br>' +
                             'Distance: %{marker.color:.2f}m<extra></extra>'
            ))
        
        # Add Radar point cloud
        if len(radar_points) > 0:
            fig.add_trace(go.Scatter3d(
                x=radar_points[:, 0],
                y=radar_points[:, 1],
                z=radar_points[:, 2],
                mode='markers',
                marker=dict(
                    size=1,
                    color='red',
                    symbol='diamond',
                    opacity=0.8
                ),
                name=f'Radar ({len(radar_points):,} points)',
                hovertemplate='<b>Radar Point</b><br>' +
                             'X: %{x:.2f}m<br>' +
                             'Y: %{y:.2f}m<br>' + 
                             'Z: %{z:.2f}m<br><extra></extra>'
            ))
        
        # Add bounding boxes
        for i, ann in enumerate(annotations):
            center = np.array(ann['translation'])
            size = ann['size']
            
            # Simplified box corners
            corners = get_simple_box_corners(center, size)
            
            # Draw 12 edges of the cube
            edges = [
                [0,1], [1,2], [2,3], [3,0],  # Bottom face
                [4,5], [5,6], [6,7], [7,4],  # Top face  
                [0,4], [1,5], [2,6], [3,7]   # Vertical edges
            ]
            
            for edge in edges:
                start, end = corners[edge[0]], corners[edge[1]]
                fig.add_trace(go.Scatter3d(
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    z=[start[2], end[2]], 
                    mode='lines',
                    line=dict(color='orange', width=4),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add labels
            fig.add_trace(go.Scatter3d(
                x=[center[0]],
                y=[center[1]], 
                z=[center[2] + size[2]/2 + 1],
                mode='text',
                text=[ann['category_name']],
                textfont=dict(size=12, color='black'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add ego vehicle
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=10, color='blue', symbol='circle'),
            name='EGO Vehicle',
            hovertemplate='<b>EGO Vehicle</b><br>Origin (0,0,0)<extra></extra>'
        ))
        
        # Set layout
        fig.update_layout(
            title=f'TruckScenes Interactive 3D Visualization<br>Sample: {sample_token[:16]}...',
            scene=dict(
                xaxis_title='X (Forward) [m]',
                yaxis_title='Y (Left) [m]', 
                zaxis_title='Z (Up) [m]',
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1200,
            height=800,
            showlegend=True
        )
        
        # Save HTML file
        fig.write_html(save_path)
        print(f"✓ Saved interactive 3D HTML: {os.path.basename(save_path)}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating interactive 3D HTML: {e}")
        return False


def get_simple_box_corners(center, size):
    """Get 8 corners of a simple cube"""
    w, l, h = size
    corners = np.array([
        [-l/2, -w/2, -h/2], [+l/2, -w/2, -h/2],
        [+l/2, +w/2, -h/2], [-l/2, +w/2, -h/2],
        [-l/2, -w/2, +h/2], [+l/2, -w/2, +h/2],
        [+l/2, +w/2, +h/2], [-l/2, +w/2, +h/2]
    ])
    return corners + center


def save_multi_view_visualizations(lidar_points, radar_points, annotations, 
                                  sample_token, out_dir, max_distance=50.0):
    """
    Save multi-view visualization images
    
    Args:
        lidar_points: LiDAR point cloud data
        radar_points: Radar point cloud data
        annotations: Bounding boxes
        sample_token: Sample token
        out_dir: Output directory
        max_distance: Maximum display distance
    
    Returns:
        dict: Saved image paths
    """
    saved_views = {}
    
    # Prepare data
    plt.style.use('default')  # Use default style
    
    # Define camera parameters for three views
    views = {
        'side_view': {'elev': 25, 'azim': 45, 'title': 'Side View'},
        'front_view': {'elev': 0, 'azim': 0, 'title': 'Front View (Driver Perspective)'},
        'top_view': {'elev': 90, 'azim': 0, 'title': 'Bird\'s Eye View (Top Down)'}
    }
    
    for view_name, view_params in views.items():
        try:
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot LiDAR point cloud
            total_lidar_points = 0
            if len(lidar_points) > 0:
                # Distance filtering
                distances = np.linalg.norm(lidar_points[:, :2], axis=1)
                mask = distances <= max_distance
                filtered_lidar = lidar_points[mask]
                
                if len(filtered_lidar) > 0:
                    # Downsample
                    step = max(1, len(filtered_lidar) // 30000)
                    viz_lidar = filtered_lidar[::step]
                    
                    # Color by distance
                    lidar_distances = np.linalg.norm(viz_lidar[:, :2], axis=1)
                    scatter = ax.scatter(viz_lidar[:, 0], viz_lidar[:, 1], viz_lidar[:, 2],
                                       s=0.8, c=lidar_distances, cmap='viridis', 
                                       alpha=0.7, label=f'LiDAR ({len(viz_lidar):,} points)')
                    total_lidar_points = len(viz_lidar)
            
            # Plot Radar point cloud
            total_radar_points = 0
            if len(radar_points) > 0:
                distances = np.linalg.norm(radar_points[:, :2], axis=1)
                mask = distances <= max_distance
                filtered_radar = radar_points[mask]
                
                if len(filtered_radar) > 0:
                    ax.scatter(filtered_radar[:, 0], filtered_radar[:, 1], filtered_radar[:, 2],
                              s=25, c='red', alpha=0.9, marker='^', 
                              label=f'Radar ({len(filtered_radar):,} points)')
                    total_radar_points = len(filtered_radar)
            
            # Plot 3D bounding boxes
            box_count = 0
            for ann in annotations:
                center = np.array(ann['translation'])
                distance = np.linalg.norm(center[:2])
                
                if distance <= max_distance * 1.2:
                    # Simplified box drawing
                    size = ann['size']
                    corners = get_simple_box_corners(center, size)
                    
                    # Draw 12 edges
                    edges = [
                        [0,1], [1,2], [2,3], [3,0],  # Bottom face
                        [4,5], [5,6], [6,7], [7,4],  # Top face
                        [0,4], [1,5], [2,6], [3,7]   # Vertical edges
                    ]
                    
                    for edge in edges:
                        start, end = corners[edge[0]], corners[edge[1]]
                        ax.plot3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                                 color='orange', linewidth=2, alpha=0.8)
                    
                    # Add labels
                    ax.text(center[0], center[1], center[2] + size[2]/2 + 1,
                           ann['category_name'], fontsize=8, ha='center')
                    box_count += 1
            
            # Plot ego vehicle
            ax.scatter([0], [0], [0], s=200, c='blue', marker='o', 
                      label='EGO', alpha=1.0)
            
            # Set view angle
            ax.view_init(elev=view_params['elev'], azim=view_params['azim'])
            
            # Set axis labels and title
            ax.set_xlabel('X (Forward) [m]', fontsize=12)
            ax.set_ylabel('Y (Left) [m]', fontsize=12)
            ax.set_zlabel('Z (Up) [m]', fontsize=12)
            
            title = (f'{view_params["title"]} - TruckScenes Sensor Fusion\n'
                    f'Sample: {sample_token[:16]}...\n'
                    f'LiDAR: {total_lidar_points:,} | Radar: {total_radar_points:,} | Boxes: {box_count}')
            ax.set_title(title, fontsize=14, pad=20)
            
            # Set axis range
            if view_name == 'top_view':
                # Bird's-eye view: adjust Z-axis range to highlight top-down effect
                ax.set_xlim([max_distance, -max_distance])
                ax.set_ylim([max_distance, -max_distance])
                ax.set_zlim([-2, 8])
            elif view_name == 'front_view':
                # Front view: focus on the front area
                ax.set_xlim([0, max_distance])
                ax.set_ylim([max_distance/2, -max_distance/2])
                ax.set_zlim([-3, 10])
            else:
                # Side view: keep original range
                ax.set_xlim([-max_distance, max_distance])
                ax.set_ylim([-max_distance, max_distance])
                ax.set_zlim([-5, 15])
            
            # Add color bar (only for images with LiDAR data)
            if total_lidar_points > 0 and 'scatter' in locals():
                cbar = plt.colorbar(scatter, ax=ax, label='Distance [m]', shrink=0.6)
                cbar.ax.tick_params(labelsize=10)
            
            # Add legend
            ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10)
            
            # Set grid
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save image
            img_path = os.path.join(out_dir, f"{view_name}.png")
            plt.savefig(img_path, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            saved_views[view_name] = img_path
            print(f"✓ Saved {view_params['title']}: {os.path.basename(img_path)}")
            
        except Exception as e:
            print(f"❌ Error saving {view_name}: {e}")
    
    return saved_views


def save_camera_config(out_dir, max_distance=50.0):
    """Save camera configuration parameters for future use"""
    camera_config = {
        "views": {
            "side_view": {
                "elevation": 25,
                "azimuth": 45,
                "description": "General overview from side angle"
            },
            "front_view": {
                "elevation": 0,
                "azimuth": 0,
                "description": "Driver perspective looking forward"
            },
            "top_view": {
                "elevation": 90,
                "azimuth": 0, 
                "description": "Bird's eye view from above"
            }
        },
        "display_limits": {
            "max_distance": max_distance,
            "x_range": [-max_distance, max_distance],
            "y_range": [-max_distance, max_distance],
            "z_range": [-5, 15]
        },
        "rendering_settings": {
            "lidar_point_size": 0.8,
            "radar_point_size": 25,
            "downsample_threshold": 30000,
            "figure_size": [16, 12],
            "dpi": 150
        }
    }
    
    config_path = os.path.join(out_dir, "camera_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(camera_config, f, indent=2)
    print(f"✓ Saved camera config: {os.path.basename(config_path)}")
    return config_path


def save_camera_images(visualizer, sample_token, out_dir):
    """
    Save camera images using pure API approach
    """
    saved_cameras = {}
    
    try:
        print(f"📷 === Camera Images via API only ===")
        
        # Get TruckScenes object (using official tutorial naming convention)
        nusc = None
        if hasattr(visualizer, 'nusc'):
            nusc = visualizer.nusc
        elif hasattr(visualizer, 'ts'):
            nusc = visualizer.ts
        elif hasattr(visualizer, 'truckscenes'):
            nusc = visualizer.truckscenes
        elif hasattr(visualizer, 'dataset'):
            nusc = visualizer.dataset
        else:
            # Try to find TruckScenes object from other attributes of visualizer
            for attr_name in dir(visualizer):
                if not attr_name.startswith('_'):
                    attr_val = getattr(visualizer, attr_name)
                    if hasattr(attr_val, 'get') and hasattr(attr_val, 'sample'):
                        nusc = attr_val
                        break
        
        if nusc is None:
            print("❌ Cannot find TruckScenes object in visualizer")
            print(f"Available attributes: {[attr for attr in dir(visualizer) if not attr.startswith('_')]}")
            return saved_cameras
        
        # Get sample data
        sample = nusc.get('sample', sample_token)
        
        # TruckScenes camera channel mapping
        camera_types = {
            'CAMERA_LEFT_FRONT': 'front_left_camera',
            'CAMERA_RIGHT_FRONT': 'front_right_camera', 
            'CAMERA_LEFT_BACK': 'back_left_camera',
            'CAMERA_RIGHT_BACK': 'back_right_camera'
        }
        
        print(f"   Available cameras in sample: {[k for k in sample['data'].keys() if 'CAMERA' in k]}")
        
        for cam_channel, output_name in camera_types.items():
            try:
                if cam_channel not in sample['data']:
                    print(f"   ⚠️ {cam_channel} not available")
                    continue
                
                # Get camera data through API
                cam_token = sample['data'][cam_channel]
                cam_data = nusc.get('sample_data', cam_token)
                
                # Build image path
                img_filename = cam_data['filename']
                img_path = os.path.join(nusc.dataroot, img_filename)
                
                if os.path.exists(img_path):
                    import shutil
                    original_ext = os.path.splitext(img_filename)[1] or '.jpg'
                    output_img_path = os.path.join(out_dir, f"{output_name}{original_ext}")
                    shutil.copy2(img_path, output_img_path)
                    
                    saved_cameras[cam_channel] = output_img_path
                    print(f"   ✓ Saved {cam_channel}: {output_name}{original_ext}")
                else:
                    print(f"   ❌ Image not found: {img_path}")
                    
            except Exception as e:
                print(f"   ❌ Error processing {cam_channel}: {e}")
        
        return saved_cameras
        
    except Exception as e:
        print(f"❌ Error in camera processing: {e}")
        return saved_cameras


# Update process_sample_enhanced function to ensure correct camera save function is called
def process_sample_enhanced(visualizer, scene_name, sample_token, output_root, max_distance=50.0):
    """Enhanced sample processing function"""
    try:
        print(f"\n🔄 Enhanced Processing: {scene_name} | {sample_token[:16]}...")
        
        # Check method signature
        sig = inspect.signature(visualizer.visualize_sample)
        
        if 'return_data' in sig.parameters:
            result = visualizer.visualize_sample(
                sample_token, 
                max_distance=max_distance, 
                return_data=True
            )
        else:
            print("⚠️ Using compatibility mode for sample processing...")
            result = _process_sample_compatible(visualizer, sample_token, max_distance)
        
        # Create output directory structure
        out_dir = os.path.join(output_root, scene_name, sample_token)
        ensure_dir(out_dir)
        
        # Create subdirectories
        viz_dir = os.path.join(out_dir, "visualizations")
        data_dir = os.path.join(out_dir, "data")
        ensure_dir(viz_dir)
        ensure_dir(data_dir)
        
        # 1. Save camera images using pure API approach
        camera_images = save_camera_images(visualizer, sample_token, out_dir)
        
        # 2. Save multi-view visualization images
        view_paths = save_multi_view_visualizations(
            result["merged_lidar"], result["merged_radar"], result["annotations"],
            sample_token, viz_dir, max_distance
        )
        
        # 3. Create interactive 3D HTML
        html_path = os.path.join(viz_dir, "interactive_3d.html")
        create_interactive_3d_html(
            result["merged_lidar"], result["merged_radar"], result["annotations"],
            sample_token, html_path
        )
        
        # 4. Save camera configuration
        camera_config_path = save_camera_config(viz_dir, max_distance)
        
        # 5. Save multi-format point cloud data
        lidar_files = {}
        radar_files = {}
        
        if result.get("merged_lidar") is not None and len(result["merged_lidar"]) > 0:
            # Calculate LiDAR point colors (based on distance)
            distances = np.linalg.norm(result["merged_lidar"][:, :2], axis=1)
            # Normalize distances to 0-1, then map to colors
            norm_distances = distances / max_distance
            colors = plt.cm.viridis(norm_distances)[:, :3]  # RGB colors
            
            lidar_base = os.path.join(data_dir, "merged_lidar")
            lidar_files = save_point_cloud_multi_format(
                result["merged_lidar"], lidar_base, colors
            )
        
        if result.get("merged_radar") is not None and len(result["merged_radar"]) > 0:
            # Radar points use red color
            radar_colors = np.tile([1.0, 0.0, 0.0], (len(result["merged_radar"]), 1))
            
            radar_base = os.path.join(data_dir, "merged_radar")
            radar_files = save_point_cloud_multi_format(
                result["merged_radar"], radar_base, radar_colors
            )
        
        # 6. Save statistics and annotations
        stats_path = os.path.join(out_dir, "stats.json")
        enhanced_stats = result["stats"].copy()
        enhanced_stats.update({
            "view_paths": view_paths,
            "interactive_3d_path": html_path,
            "camera_images": camera_images,
            "lidar_files": lidar_files,
            "radar_files": radar_files,
            "camera_config_path": camera_config_path,
            "output_structure": {
                "visualizations": viz_dir,
                "data": data_dir,
                "main": out_dir
            }
        })
        save_stats(enhanced_stats, stats_path)
        
        # Save annotation information
        if result.get("annotations"):
            ann_path = os.path.join(data_dir, "annotations.json")
            save_annotations(result["annotations"], ann_path)
        
        # 7. Create README file
        readme_path = os.path.join(out_dir, "README.md")
        create_sample_readme(enhanced_stats, readme_path, scene_name, sample_token)
        
        print(f"✅ Enhanced processing completed: {scene_name} | {sample_token[:16]}")
        print(f"   📁 Output directory: {out_dir}")
        print(f"   📷 Camera images: {len(camera_images)}")
        print(f"   🖼️  Views saved: {len(view_paths)}")
        print(f"   📊 Interactive 3D: {os.path.exists(html_path)}")
        print(f"   💾 Point cloud formats: LiDAR({len(lidar_files)}), Radar({len(radar_files)})")
        
        return enhanced_stats
        
    except Exception as e:
        error_msg = f"Enhanced processing error {scene_name} | {sample_token}: {str(e)}"
        print(f"❌ {error_msg}")
        return None


def _process_sample_compatible(visualizer, sample_token, max_distance):
    """Compatible processing for older versions"""
    try:
        visualizer.visualize_sample(sample_token, max_distance=max_distance)
        
        return {
            "merged_lidar": np.array([]),
            "merged_radar": np.array([]),
            "annotations": [],
            "stats": {
                "lidar_points": 0,
                "radar_points": 0,
                "box_count": 0,
                "max_distance": max_distance,
                "compatible_mode": True
            }
        }
    except Exception as e:
        print(f"❌ Compatible mode failed: {e}")
        raise e


def save_stats(stats, save_path):
    """
    Save statistics to JSON file
    
    Args:
        stats: Statistics dictionary
        save_path: Save path
    """
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        print(f"✓ Saved stats: {os.path.basename(save_path)}")
        return True
    except Exception as e:
        print(f"❌ Error saving stats {save_path}: {e}")
        return False


def save_annotations(annotations, save_path):
    """
    Save annotation information to JSON file
    
    Args:
        annotations: Annotation information list
        save_path: Save path
    """
    try:
        # Ensure annotation data can be serialized
        serializable_annotations = []
        for ann in annotations:
            ann_copy = {}
            for key, value in ann.items():
                if isinstance(value, np.ndarray):
                    ann_copy[key] = value.tolist()
                else:
                    ann_copy[key] = value
            serializable_annotations.append(ann_copy)
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(serializable_annotations, f, indent=2, ensure_ascii=False, default=str)
        print(f"✓ Saved annotations: {os.path.basename(save_path)} ({len(annotations)} boxes)")
        return True
    except Exception as e:
        print(f"❌ Error saving annotations {save_path}: {e}")
        return False


def create_sample_readme(stats, readme_path, scene_name, sample_token):
    """
    Create README file for sample processing results
    
    Args:
        stats: Statistics dictionary
        readme_path: README file save path
        scene_name: Scene name
        sample_token: Sample token
    """
    try:
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# TruckScenes Sample: {sample_token[:16]}\n\n")
            f.write(f"**Scene:** {scene_name}  \n")
            f.write(f"**Sample Token:** {sample_token}  \n")
            f.write(f"**Processed At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
            
            f.write(f"## 📊 Statistics\n\n")
            f.write(f"- **LiDAR Points:** {stats.get('lidar_points', 0):,}\n")
            f.write(f"- **Radar Points:** {stats.get('radar_points', 0):,}\n")
            f.write(f"- **Annotation Boxes:** {stats.get('box_count', 0)}\n")
            f.write(f"- **Camera Images:** {len(stats.get('camera_images', {}))}\n")
            f.write(f"- **Max Distance:** {stats.get('max_distance', 50.0)}m\n\n")
            
            f.write(f"## 📁 File Structure\n\n")
            f.write(f"```\n")
            f.write(f"{sample_token[:16]}/\n")
            f.write(f"├── front_camera.jpg            # Front camera image\n")
            f.write(f"├── front_left_camera.jpg       # Front left camera image\n")
            f.write(f"├── front_right_camera.jpg      # Front right camera image\n")
            f.write(f"├── back_camera.jpg             # Back camera image\n")
            f.write(f"├── visualizations/\n")
            f.write(f"│   ├── side_view.png           # Side view\n")
            f.write(f"│   ├── front_view.png          # Front view\n")
            f.write(f"│   ├── top_view.png            # Bird's eye view\n")
            f.write(f"│   └── interactive_3d.html     # Interactive 3D visualization\n")
            f.write(f"├── data/\n")
            f.write(f"│   ├── merged_lidar.*          # LiDAR point cloud data\n")
            f.write(f"│   ├── merged_radar.*          # Radar point cloud data\n")
            f.write(f"│   └── annotations.json        # 3D bounding boxes\n")
            f.write(f"├── stats.json                  # Statistics information\n")
            f.write(f"└── README.md                   # This file\n")
            f.write(f"```\n\n")
            
            f.write(f"## 📷 Camera Images\n\n")
            if 'camera_images' in stats:
                for cam_channel, img_path in stats['camera_images'].items():
                    img_name = os.path.basename(img_path)
                    f.write(f"- **{cam_channel}:** `{img_name}`\n")
            f.write(f"\n")
            
            f.write(f"## 🚀 Quick Start\n\n")
            f.write(f"1. **View Scene:** Double-click `front_camera.jpg` and other image files\n")
            f.write(f"2. **3D Visualization:** Open `visualizations/interactive_3d.html`\n")
            f.write(f"3. **Comparative Analysis:** Compare camera images with 3D visualizations to understand the scene\n\n")
            
            f.write(f"---\n")
            f.write(f"*Generated by TruckScenes Output Management System*\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating README {readme_path}: {e}")
        return False


def create_sample_config():
    """Create sample configuration file"""
    sample_config = {
        "scene-0001": [
            "32d2bcf46e734dffb14fe2e0a823d059",
            "e3b2c1d4a5f6789012345678901234ab"
        ],
        "scene-0002": [
            "f4c3b2a1d5e6789012345678901234cd",
            "a1b2c3d4e5f6789012345678901234ef"
        ]
    }
    
    with open(SELECTED_JSON, "w", encoding="utf-8") as f:
        json.dump(sample_config, f, indent=2)
    print(f"✓ Created sample config: {SELECTED_JSON}")


def generate_summary_report(all_stats, summary_dir):
    """Generate summary report"""
    if not all_stats:
        print("⚠️ No data available for summary report")
        return
    
    df = pd.DataFrame(all_stats)
    
    # Save complete statistics CSV
    csv_path = os.path.join(summary_dir, "all_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved complete stats: {csv_path}")
    
    # Generate summary statistics
    summary_stats = {
        "total_samples": len(df),
        "total_scenes": df['scene_name'].nunique() if 'scene_name' in df.columns else 0,
        "total_lidar_points": int(df['lidar_points'].sum()) if 'lidar_points' in df.columns else 0,
        "total_radar_points": int(df['radar_points'].sum()) if 'radar_points' in df.columns else 0,
        "total_boxes": int(df['box_count'].sum()) if 'box_count' in df.columns else 0,
        "avg_lidar_points": float(df['lidar_points'].mean()) if 'lidar_points' in df.columns else 0,
        "avg_radar_points": float(df['radar_points'].mean()) if 'radar_points' in df.columns else 0,
        "avg_boxes": float(df['box_count'].mean()) if 'box_count' in df.columns else 0,
        "processing_date": datetime.now().isoformat()
    }
    
    # Save summary statistics
    summary_path = os.path.join(summary_dir, "summary_stats.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"✓ Saved summary stats: {summary_path}")
    
    # Group statistics by scene
    if 'scene_name' in df.columns:
        scene_stats = df.groupby('scene_name').agg({
            'lidar_points': ['count', 'sum', 'mean'] if 'lidar_points' in df.columns else ['count'],
            'radar_points': ['sum', 'mean'] if 'radar_points' in df.columns else [],
            'box_count': ['sum', 'mean'] if 'box_count' in df.columns else []
        }).round(2)
        
        scene_csv_path = os.path.join(summary_dir, "scene_summary.csv")
        scene_stats.to_csv(scene_csv_path)
        print(f"✓ Saved scene summary: {scene_csv_path}")
    
    print(f"\n📊 Processing Summary:")
    print(f"  - Total Samples: {summary_stats['total_samples']}")
    print(f"  - Total Scenes: {summary_stats['total_scenes']}")
    print(f"  - Total LiDAR Points: {summary_stats['total_lidar_points']:,}")
    print(f"  - Total Radar Points: {summary_stats['total_radar_points']:,}")
    print(f"  - Total Annotation Boxes: {summary_stats['total_boxes']}")


def main():
    """Main function"""
    print("🚛 TruckScenes Batch Output Management System")
    print("=" * 50)
    
    # ==== 4. Check configuration file ====
    if not os.path.exists(SELECTED_JSON):
        print(f"⚠️ Configuration file does not exist: {SELECTED_JSON}")
        print("🔧 Creating sample configuration file...")
        create_sample_config()
        print("📝 Please edit the configuration file and add scenes and sample tokens to process")
        return
    
    # Load configuration
    try:
        with open(SELECTED_JSON, "r", encoding="utf-8") as f:
            scene_samples = json.load(f)
        print(f"✓ Loaded configuration file: {len(scene_samples)} scenes")
    except Exception as e:
        print(f"❌ Unable to load configuration file: {e}")
        return
    
    # ==== 5. Initialize visualization tool ====
    try:
        print("🔧 Initializing TruckScenes visualization tool...")
        visualizer = TruckScenesVisualizer(DATA_DIR, version="v1.0-mini")
        print("✓ Visualization tool initialized successfully")
    except Exception as e:
        print(f"❌ Unable to initialize visualization tool: {e}")
        return
    
    # ==== 6. Batch processing ====
    all_stats = []
    total_samples = sum(len(samples) for samples in scene_samples.values())
    processed_count = 0
    
    print(f"\n🚀 Starting batch processing of {total_samples} samples...")
    
    for scene_name, sample_list in scene_samples.items():
        print(f"\n📂 Processing scene: {scene_name} ({len(sample_list)} samples)")
        
        for sample_token in sample_list:
            processed_count += 1
            print(f"[{processed_count}/{total_samples}] ", end="")
            
            stats = process_sample_enhanced(
                visualizer, scene_name, sample_token, OUTPUT_ROOT
            )
            
            if stats is not None:
                stats["scene_name"] = scene_name
                stats["sample_token"] = sample_token
                all_stats.append(stats)
    
    # ==== 7. Generate summary report ====
    print(f"\n📈 Generating summary report...")
    generate_summary_report(all_stats, SUMMARY_DIR)
    
    # ==== 8. Completion information ====
    success_count = len(all_stats)
    failure_count = total_samples - success_count
    
    print(f"\n🎉 Batch processing completed!")
    print(f"  ✅ Success: {success_count}/{total_samples}")
    if failure_count > 0:
        print(f"  ❌ Failed: {failure_count}/{total_samples}")
    print(f"  📁 Output directory: {OUTPUT_ROOT}")
    print(f"  📊 Summary report: {SUMMARY_DIR}")
    
    if success_count == 0:
        print("\n⚠️ No samples were successfully processed, please check:")
        print("  1. Data path is correct")
        print("  2. Sample tokens are valid")
        print("  3. Dependencies are correctly installed")


if __name__ == "__main__":
    main()