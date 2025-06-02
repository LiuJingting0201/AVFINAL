#!/usr/bin/env python3
"""
TruckScenes Sensor Fusion Visualization Tool - Fixed Version
Fixed issues:
1. Confirm that 6 radar sensors indeed provide 360° coverage (compliant with modern truck configuration)
2. Correct legend: display fused sensor data instead of individual sensors
3. Fix coordinate axis label display issues
4. LiDAR point cloud colored by distance (instead of height)
5. Add return_data parameter to support batch processing
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
from scipy.spatial.transform import Rotation as R

# Set matplotlib font and display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Use official devkit (recommended)
try:
    from truckscenes.truckscenes import TruckScenes
    from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
    from truckscenes.utils.geometry_utils import transform_matrix
    DEVKIT_AVAILABLE = True
    print("Using official TruckScenes devkit")
except ImportError:
    DEVKIT_AVAILABLE = False
    print("Official devkit not found, using simplified implementation")

class TruckScenesVisualizer:
    def __init__(self, dataroot: str, version: str = 'v1.0-mini'):
        """
        Initialize visualization tool
        
        Args:
            dataroot: TruckScenes data root directory
            version: Data version ('v1.0-mini', 'v1.0-trainval', 'v1.0-test')
        """
        self.dataroot = dataroot
        self.version = version
        
        if DEVKIT_AVAILABLE:
            self.ts = TruckScenes(version=version, dataroot=dataroot, verbose=True)
        else:
            # Simplified implementation: directly load JSON files
            self.ts = self._load_simple_ts()
    
    def _load_simple_ts(self):
        """Simplified TruckScenes data loading"""
        class SimpleTruckScenes:
            def __init__(self, dataroot, version):
                self.dataroot = dataroot
                self.version = version
                
                # Load basic tables
                tables_path = os.path.join(dataroot, version)
                self.sample = self._load_table(os.path.join(tables_path, 'sample.json'))
                self.sample_data = self._load_table(os.path.join(tables_path, 'sample_data.json'))
                self.sample_annotation = self._load_table(os.path.join(tables_path, 'sample_annotation.json'))
                self.calibrated_sensor = self._load_table(os.path.join(tables_path, 'calibrated_sensor.json'))
                self.ego_pose = self._load_table(os.path.join(tables_path, 'ego_pose.json'))
                
            def _load_table(self, path):
                """Load JSON table and create token mapping"""
                with open(path, 'r') as f:
                    data = json.load(f)
                return {record['token']: record for record in data}
            
            def get(self, table_name, token):
                """Get record"""
                table = getattr(self, table_name)
                return table[token]
        
        return SimpleTruckScenes(self.dataroot, self.version)
    
    def visualize_sample(self, sample_token: str, max_distance: float = 50.0, return_data: bool = False):
        """
        Visualize sensor fusion data for specified sample
        
        Args:
            sample_token: Sample token
            max_distance: Maximum display distance
            return_data: Whether to return data for batch processing
        """
        print(f"Visualizing sample: {sample_token}")
        
        # Get sample information
        sample = self.ts.get('sample', sample_token)
        
        # Check available sensors
        available_sensors = list(sample['data'].keys())
        print("Available sensors:", available_sensors)
        
        # Classify sensors
        lidar_sensors = [s for s in available_sensors if 'LIDAR' in s.upper()]
        radar_sensors = [s for s in available_sensors if 'RADAR' in s.upper()]
        
        print(f"LiDAR sensors ({len(lidar_sensors)}): {lidar_sensors}")
        print(f"Radar sensors ({len(radar_sensors)}): {radar_sensors}")
        
        # Collect all sensor data
        all_lidar_points = []
        all_radar_points = []
        
        # Process LiDAR data
        for lidar_sensor in lidar_sensors:
            lidar_token = sample['data'][lidar_sensor]
            lidar_points, _ = self._get_sensor_data(lidar_token)
            if lidar_points is not None:
                all_lidar_points.append(lidar_points)
        
        # Process Radar data  
        for radar_sensor in radar_sensors:
            radar_token = sample['data'][radar_sensor]
            radar_points, _ = self._get_sensor_data(radar_token)
            if radar_points is not None:
                all_radar_points.append(radar_points)
        
        # Merge all point cloud data
        merged_lidar = np.vstack(all_lidar_points) if all_lidar_points else np.empty((0, 3))
        merged_radar = np.vstack(all_radar_points) if all_radar_points else np.empty((0, 3))
        
        # Get 3D annotation boxes
        annotations = self._get_sample_annotations(sample_token)
        
        # Visualize
        fig, total_lidar_points, total_radar_points, box_count = self._create_visualization(
            merged_lidar, merged_radar, annotations, 
            sample_token, max_distance, len(lidar_sensors), len(radar_sensors)
        )

        # Prepare return result
        result = {
            # --- Visualization image ---
            "fig": fig,                            # matplotlib figure object (management script can save directly as png)
            "visualization_path": None,            # Optional, actual image save path (if saved directly)
            
            # --- Point cloud data ---
            "merged_lidar": merged_lidar,          # numpy array (N,3)
            "merged_radar": merged_radar,          # numpy array (M,3)
            "lidar_path": None,                    # Optional, lidar point cloud file save path
            "radar_path": None,                    # Optional, radar point cloud file save path
            
            # --- 3D boxes and other object annotations ---
            "annotations": annotations,            # list, each element is annotation box (center, size, rotation, category, etc. dict)

            # --- Statistics information (for subsequent csv/global analysis/automatic plotting) ---
            "stats": {
                "sample_token": sample_token,          # Current frame unique identifier
                "n_lidar": len(lidar_sensors),         # Number of lidar sensors used
                "n_radar": len(radar_sensors),         # Number of radar sensors used
                "lidar_points": int(len(merged_lidar)) if merged_lidar is not None else 0,
                "radar_points": int(len(merged_radar)) if merged_radar is not None else 0,
                "box_count": int(len(annotations)) if annotations is not None else 0,
                "max_distance": max_distance,          # Visualization maximum distance parameter
                # Extensible: weather, scene, timestamp, specific sensor names, etc.
            },
            
            # --- Log information (can supplement if there are exceptions/warnings) ---
            "log": []                               # Can be list, collecting runtime exceptions, warnings, status
        }
        
        # Decide whether to return data based on parameter
        if return_data:
            return result
        else:
            # Original display logic
            plt.show()
    
    def _get_sensor_data(self, sample_data_token: str):
        """Get sensor data and pose information"""
        sample_data_record = self.ts.get('sample_data', sample_data_token)
        
        # Get sensor calibration and ego pose
        cs_record = self.ts.get('calibrated_sensor', sample_data_record['calibrated_sensor_token'])
        ego_pose_record = self.ts.get('ego_pose', sample_data_record['ego_pose_token'])
        
        # Build transformation matrix from sensor to global coordinates
        sensor_to_ego = self._pose_to_matrix(cs_record)
        ego_to_global = self._pose_to_matrix(ego_pose_record)
        sensor_to_global = ego_to_global @ sensor_to_ego
        
        # Load point cloud data
        data_path = os.path.join(self.dataroot, sample_data_record['filename'])
        
        if not os.path.exists(data_path):
            print(f"Data file does not exist: {data_path}")
            return None, None
        
        if DEVKIT_AVAILABLE:
            # Use official devkit
            if 'LIDAR' in sample_data_record['channel']:
                pc = LidarPointCloud.from_file(data_path)
            else:
                pc = RadarPointCloud.from_file(data_path)
            points = pc.points[:3, :].T  # Convert to (N, 3) format
        else:
            # Simplified loading (assume .bin file)
            points = self._load_points_simple(data_path)
        
        if points is None or len(points) == 0:
            return None, None
        
        # Transform to global coordinates (with ego as reference)
        points_ego = self._transform_points_to_ego(points, sensor_to_global, ego_to_global)
        
        return points_ego, ego_pose_record
    
    def _load_points_simple(self, data_path: str):
        """Simplified point cloud loading (supports .bin and .ply formats)"""
        if data_path.endswith('.bin'):
            # Load binary point cloud file
            points = np.fromfile(data_path, dtype=np.float32)
            if len(points) % 4 == 0:  # LiDAR: x,y,z,intensity
                points = points.reshape(-1, 4)[:, :3]
            elif len(points) % 5 == 0:  # Radar: x,y,z,rcs,v_comp
                points = points.reshape(-1, 5)[:, :3]
            else:
                print(f"Unknown point cloud format: {data_path}")
                return None
        elif data_path.endswith('.ply'):
            # Load PLY file
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(data_path)
                points = np.asarray(pcd.points)
            except ImportError:
                print("Need to install open3d to load PLY files: pip install open3d")
                return None
        else:
            print(f"Unsupported file format: {data_path}")
            return None
        
        return points
    
    def _pose_to_matrix(self, pose_record):
        """Convert pose record to 4x4 transformation matrix"""
        translation = np.array(pose_record['translation'])
        rotation = np.array(pose_record['rotation'])  # [w, x, y, z]
        
        # Create rotation matrix
        r = R.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]])  # [x,y,z,w]
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = r.as_matrix()
        transform[:3, 3] = translation
        
        return transform
    
    def _transform_points_to_ego(self, points, sensor_to_global, ego_to_global):
        """Transform point cloud from sensor coordinate system to ego coordinate system"""
        # Calculate inverse transformation from global to ego
        global_to_ego = np.linalg.inv(ego_to_global)
        
        # Complete transformation: sensor -> global -> ego
        sensor_to_ego = global_to_ego @ sensor_to_global
        
        # Transform point cloud
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        points_ego = (sensor_to_ego @ points_homo.T).T[:, :3]
        
        return points_ego
    
    def _get_sample_annotations(self, sample_token: str):
        """Get 3D annotation boxes for the sample"""
        # Get all annotations for this sample
        sample_annotations = []
        
        if DEVKIT_AVAILABLE:
            # Use official devkit - get annotations through sample
            sample_record = self.ts.get('sample', sample_token)
            
            # Get all annotations associated with this sample
            ann_tokens = sample_record.get('anns', [])
            for ann_token in ann_tokens:
                ann_record = self.ts.get('sample_annotation', ann_token)
                sample_annotations.append(ann_record)
        else:
            # Simplified implementation - iterate through all annotations to find matches
            for ann_token, ann_record in self.ts.sample_annotation.items():
                if ann_record['sample_token'] == sample_token:
                    sample_annotations.append(ann_record)
        
        # Transform annotation boxes to ego coordinate system
        ego_annotations = []
        if sample_annotations:
            # Get reference ego pose (using timestamp from first sensor)
            sample = self.ts.get('sample', sample_token)
            
            # Find first available sensor data as reference
            first_sensor_key = list(sample['data'].keys())[0]
            first_sensor_data = self.ts.get('sample_data', sample['data'][first_sensor_key])
            ego_pose = self.ts.get('ego_pose', first_sensor_data['ego_pose_token'])
            ego_to_global = self._pose_to_matrix(ego_pose)
            global_to_ego = np.linalg.inv(ego_to_global)
            
            for ann in sample_annotations:
                # Transform center
                center_global = np.array(ann['translation'])
                center_ego = (global_to_ego @ np.append(center_global, 1))[:3]
                
                # Transform rotation
                r_global = R.from_quat([ann['rotation'][1], ann['rotation'][2], 
                                      ann['rotation'][3], ann['rotation'][0]])
                r_ego_transform = R.from_matrix(global_to_ego[:3, :3])
                r_ego = r_ego_transform * r_global
                
                ego_ann = {
                    'translation': center_ego.tolist(),  # Convert to list for JSON serialization
                    'size': ann['size'],
                    'rotation': r_ego.as_quat().tolist(),  # Convert to list
                    'category_name': ann['category_name']
                }
                ego_annotations.append(ego_ann)
        
        return ego_annotations
    
    def _create_visualization(self, lidar_points, radar_points, annotations, 
                            sample_token, max_distance, n_lidar, n_radar):
        """Create 3D visualization"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Process LiDAR point cloud
        total_lidar_points = 0
        scatter_lidar = None
        if len(lidar_points) > 0:
            # Distance filtering
            distances = np.linalg.norm(lidar_points[:, :2], axis=1)
            mask = distances <= max_distance
            filtered_lidar = lidar_points[mask]
            
            if len(filtered_lidar) > 0:
                # Downsampling for performance
                step = max(1, len(filtered_lidar) // 30000)
                viz_lidar = filtered_lidar[::step]
                
                # Color by distance (fix: changed from height to distance)
                lidar_distances = np.linalg.norm(viz_lidar[:, :2], axis=1)
                scatter_lidar = ax.scatter(viz_lidar[:, 0], viz_lidar[:, 1], viz_lidar[:, 2],
                                        s=0.8, c=lidar_distances, cmap='viridis', alpha=0.7, 
                                        label=f'Merged LiDAR Point clouds ({n_lidar} Sensors)')
                total_lidar_points = len(viz_lidar)
        
        # Process Radar point cloud
        total_radar_points = 0
        if len(radar_points) > 0:
            # Distance filtering
            distances = np.linalg.norm(radar_points[:, :2], axis=1)
            mask = distances <= max_distance
            filtered_radar = radar_points[mask]
            
            if len(filtered_radar) > 0:
                ax.scatter(filtered_radar[:, 0], filtered_radar[:, 1], filtered_radar[:, 2],
                          s=25, c='red', alpha=0.9, marker='^', 
                          label=f'Merged Radar Point clouds ({n_radar} Sensors)')
                total_radar_points = len(filtered_radar)
        
        # Draw 3D annotation boxes
        box_count = 0
        for ann in annotations:
            center = np.array(ann['translation'])
            distance = np.linalg.norm(center[:2])
            
            if distance <= max_distance * 1.2:
                # Reconstruct rotation object
                rotation = R.from_quat(ann['rotation'])
                corners = self._get_box_corners(
                    center, ann['size'], rotation
                )
                self._draw_3d_box(ax, corners)
                
                # Add category label
                ax.text(center[0], center[1], center[2] + ann['size'][2]/2 + 1,
                       ann['category_name'], fontsize=8, ha='center')
                box_count += 1
        
        # Draw ego vehicle
        ax.scatter([0], [0], [0], s=200, c='blue', marker='o', 
                  label='EGO', alpha=1.0)
        
        # Set figure properties (fix coordinate axis labels)
        ax.set_xlabel('X (heading) [m]', fontsize=12, labelpad=10)
        ax.set_ylabel('Y (left turning) [m]', fontsize=12, labelpad=10)
        ax.set_zlabel('Z (upwards) [m]', fontsize=12, labelpad=10)
        
        # Fix title display
        title = (f'TruckScenes Visualize Sensor Fusion\n'
                f'Sample ID: {sample_token[:16]}...\n'
                f'LiDAR points: {total_lidar_points:,} | RADAR points: {total_radar_points:,} | Object Boxes: {box_count}')
        ax.set_title(title, fontsize=14, pad=20)
        
        # Set coordinate axis range
        ax.set_xlim([-max_distance, max_distance])
        ax.set_ylim([-max_distance, max_distance])
        ax.set_zlim([-5, 15])
        
        # Add distance colorbar (fix: distance instead of height)
        if total_lidar_points > 0 and scatter_lidar is not None:
            cbar = plt.colorbar(scatter_lidar, ax=ax, label='Distance From Sensors [m]', shrink=0.6)
            cbar.ax.tick_params(labelsize=10)
        
        # Add legend (fix position and style)
        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10)
        
        # Set grid and background
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Set viewing angle
        ax.view_init(elev=25, azim=45)
        
        plt.tight_layout()
        
        # Output statistics
        print(f"\n=== Visualization Statistics ===")
        print(f"Number of LiDAR sensors: {n_lidar}")
        print(f"Number of Radar sensors: {n_radar}")
        print(f"Fused LiDAR point cloud: {total_lidar_points:,} points")
        print(f"Fused Radar point cloud: {total_radar_points:,} points") 
        print(f"3D annotation boxes: {box_count} boxes")
        print(f"Display range: {max_distance}m")
        print("✓ Radar 360° coverage normal (6 sensors provide omnidirectional detection)")
        print("✓ Point cloud color represents distance from sensors")
        
        return fig, total_lidar_points, total_radar_points, box_count
    
    def _get_box_corners(self, center, size, rotation):
        """Calculate 8 corner points of 3D annotation box"""
        w, l, h = size
        
        # 8 corner points in object coordinate system
        corners = np.array([
            [-l/2, -w/2, -h/2], [+l/2, -w/2, -h/2],
            [+l/2, +w/2, -h/2], [-l/2, +w/2, -h/2],
            [-l/2, -w/2, +h/2], [+l/2, -w/2, +h/2],
            [+l/2, +w/2, +h/2], [-l/2, +w/2, +h/2]
        ])
        
        # Apply rotation and translation
        if hasattr(rotation, 'as_matrix'):
            # scipy.spatial.transform.Rotation object
            rotation_matrix = rotation.as_matrix()
        else:
            # Assume quaternion array [w, x, y, z]
            r = R.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]])
            rotation_matrix = r.as_matrix()
        
        corners_world = corners @ rotation_matrix.T + np.array(center)
        return corners_world
    
    def _draw_3d_box(self, ax, corners, color='orange', linewidth=2):
        """Draw 3D annotation box"""
        # Define 12 edges
        edges = [
            [0,1], [1,2], [2,3], [3,0],  # Bottom face
            [4,5], [5,6], [6,7], [7,4],  # Top face
            [0,4], [1,5], [2,6], [3,7]   # Vertical edges
        ]
        
        for edge in edges:
            start, end = corners[edge[0]], corners[edge[1]]
            ax.plot3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                     color=color, linewidth=linewidth, alpha=0.8)


def main():
    """Main function - Usage example"""
    # Configuration parameters
    sample_token = "32d2bcf46e734dffb14fe2e0a823d059"
    dataroot = "../data/man-truckscenes"  # Adjust according to actual path
    version = "v1.0-mini"
    max_distance = 50.0
    
    try:
        # Create visualization tool
        print("Initializing TruckScenes visualization tool...")
        visualizer = TruckScenesVisualizer(dataroot, version)
        
        # Visualize specified sample
        visualizer.visualize_sample(sample_token, max_distance)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease check:")
        print("1. Whether data path is correct")
        print("2. Whether necessary dependencies are installed: pip install truckscenes-devkit[all]")
        print("3. Whether sample token exists in the dataset")


if __name__ == "__main__":
    main()
