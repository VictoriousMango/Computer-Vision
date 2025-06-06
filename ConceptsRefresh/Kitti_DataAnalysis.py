import streamlit as st
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy.spatial.transform import Rotation as R
from scipy.stats import pearsonr
import os
import glob
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="KITTI Dataset EDA for VSLAM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


class KITTIDataAnalyzer:
    """KITTI Dataset Analyzer for Visual SLAM"""

    def __init__(self):
        # KITTI camera calibration matrices
        self.calib_cam_to_cam = {
            'P_rect_00': np.array([
                [7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00],
                [0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00],
                [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]
            ]),
            'P_rect_01': np.array([
                [7.215377e+02, 0.000000e+00, 6.095593e+02, -3.875744e+02],
                [0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00],
                [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]
            ])
        }

        # Transformation matrices
        self.T_cam0_velo = np.array([
            [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
            [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
            [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
            [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]
        ])

        # Initialize data containers
        self.sequences_data = {}
        self.current_sequence = None

    def generate_sample_data(self, sequence_id: str = "00", num_frames: int = 1000):
        """Generate sample KITTI-like data for demonstration"""
        np.random.seed(42)

        # Generate realistic trajectory (urban driving pattern)
        t = np.linspace(0, 10, num_frames)

        # Create a complex trajectory with turns and straight segments
        x_traj = 50 * np.sin(0.3 * t) + 0.1 * t * np.cos(0.5 * t)
        y_traj = np.zeros_like(t)  # Assume ground level
        z_traj = 100 * t + 20 * np.sin(0.2 * t)

        # Generate orientation (yaw, pitch, roll)
        yaw = 0.3 * t + 0.5 * np.sin(0.2 * t)
        pitch = 0.05 * np.sin(0.4 * t)
        roll = 0.02 * np.sin(0.6 * t)

        poses = []
        for i in range(num_frames):
            # Create transformation matrix
            rot = R.from_euler('xyz', [roll[i], pitch[i], yaw[i]], degrees=False)
            T = np.eye(4)
            T[:3, :3] = rot.as_matrix()
            T[:3, 3] = [x_traj[i], y_traj[i], z_traj[i]]
            poses.append(T)

        # Generate synthetic images metadata
        images_data = []
        for i in range(num_frames):
            # Simulate image quality metrics
            brightness = np.random.normal(120, 30)
            contrast = np.random.normal(50, 15)
            blur_metric = np.random.exponential(2.0)

            # Simulate feature detection results
            num_features = np.random.poisson(500) + 100
            feature_quality = np.random.beta(2, 3)

            # Simulate motion metrics
            if i > 0:
                prev_pose = poses[i - 1]
                curr_pose = poses[i]
                rel_trans = np.linalg.norm(curr_pose[:3, 3] - prev_pose[:3, 3])
                rel_rot = np.trace(prev_pose[:3, :3].T @ curr_pose[:3, :3])
                rel_rot = np.arccos(np.clip((rel_rot - 1) / 2, -1, 1))
            else:
                rel_trans = 0
                rel_rot = 0

            images_data.append({
                'frame_id': i,
                'timestamp': i * 0.1,  # 10 FPS
                'brightness': max(0, min(255, brightness)),
                'contrast': max(0, contrast),
                'blur_metric': blur_metric,
                'num_features': num_features,
                'feature_quality': feature_quality,
                'translation_speed': rel_trans / 0.1 if i > 0 else 0,
                'rotation_speed': rel_rot / 0.1 if i > 0 else 0,
                'pose': poses[i]
            })

        # Generate LiDAR data statistics
        lidar_data = []
        for i in range(num_frames):
            num_points = np.random.poisson(120000) + 50000
            max_range = np.random.normal(80, 10)
            point_density = num_points / (max_range ** 2)

            lidar_data.append({
                'frame_id': i,
                'num_points': num_points,
                'max_range': max_range,
                'point_density': point_density,
                'intensity_mean': np.random.normal(0.3, 0.1),
                'intensity_std': np.random.exponential(0.05)
            })

        return {
            'sequence_id': sequence_id,
            'poses': poses,
            'images': pd.DataFrame(images_data),
            'lidar': pd.DataFrame(lidar_data),
            'calibration': self.calib_cam_to_cam
        }

    def analyze_trajectory(self, poses: List[np.ndarray]) -> Dict:
        """Analyze trajectory characteristics"""
        if not poses:
            return {}

        positions = np.array([[pose[0, 3], pose[1, 3], pose[2, 3]] for pose in poses])

        # Calculate distances
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_distance = np.sum(distances)

        # Calculate speeds
        dt = 0.1  # 10 FPS
        speeds = distances / dt

        # Calculate accelerations
        accelerations = np.diff(speeds) / dt

        # Calculate turns (change in heading)
        headings = []
        for pose in poses:
            # Extract yaw from rotation matrix
            yaw = np.arctan2(pose[1, 0], pose[0, 0])
            headings.append(yaw)

        heading_changes = np.abs(np.diff(headings))
        # Handle angle wrapping
        heading_changes = np.minimum(heading_changes, 2 * np.pi - heading_changes)

        return {
            'total_distance': total_distance,
            'avg_speed': np.mean(speeds),
            'max_speed': np.max(speeds),
            'avg_acceleration': np.mean(np.abs(accelerations)),
            'max_acceleration': np.max(np.abs(accelerations)),
            'total_turns': np.sum(heading_changes),
            'avg_turn_rate': np.mean(heading_changes),
            'positions': positions,
            'speeds': speeds,
            'accelerations': accelerations,
            'heading_changes': heading_changes
        }

    def analyze_image_quality(self, images_df: pd.DataFrame) -> Dict:
        """Analyze image quality metrics"""
        return {
            'brightness_stats': images_df['brightness'].describe(),
            'contrast_stats': images_df['contrast'].describe(),
            'blur_stats': images_df['blur_metric'].describe(),
            'feature_count_stats': images_df['num_features'].describe(),
            'feature_quality_stats': images_df['feature_quality'].describe()
        }

    def detect_challenging_scenarios(self, images_df: pd.DataFrame, trajectory_analysis: Dict) -> Dict:
        """Detect challenging scenarios for VSLAM"""
        challenges = {
            'low_light': [],
            'low_contrast': [],
            'motion_blur': [],
            'few_features': [],
            'fast_motion': [],
            'sharp_turns': []
        }

        # Define thresholds
        low_light_thresh = 80
        low_contrast_thresh = 30
        blur_thresh = 5.0
        few_features_thresh = 200
        fast_motion_thresh = np.percentile(trajectory_analysis['speeds'], 90)
        sharp_turn_thresh = np.percentile(trajectory_analysis['heading_changes'], 90)

        for idx, row in images_df.iterrows():
            if row['brightness'] < low_light_thresh:
                challenges['low_light'].append(idx)
            if row['contrast'] < low_contrast_thresh:
                challenges['low_contrast'].append(idx)
            if row['blur_metric'] > blur_thresh:
                challenges['motion_blur'].append(idx)
            if row['num_features'] < few_features_thresh:
                challenges['few_features'].append(idx)
            if row['translation_speed'] > fast_motion_thresh:
                challenges['fast_motion'].append(idx)

        # Find sharp turns
        for i, change in enumerate(trajectory_analysis['heading_changes']):
            if change > sharp_turn_thresh:
                challenges['sharp_turns'].append(i + 1)

        return challenges


def plot_trajectory_3d(positions: np.ndarray, speeds: np.ndarray = None):
    """Create 3D trajectory plot"""
    fig = go.Figure()

    if speeds is not None:
        # Color by speed
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines+markers',
            marker=dict(
                size=4,
                color=speeds,
                colorscale='Viridis',
                colorbar=dict(title="Speed (m/s)"),
                showscale=True
            ),
            line=dict(width=4),
            name='Trajectory'
        ))
    else:
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines+markers',
            marker=dict(size=4, color='red'),
            line=dict(width=4, color='red'),
            name='Trajectory'
        ))

    # Add start and end points
    fig.add_trace(go.Scatter3d(
        x=[positions[0, 0]], y=[positions[0, 1]], z=[positions[0, 2]],
        mode='markers',
        marker=dict(size=10, color='green'),
        name='Start'
    ))

    fig.add_trace(go.Scatter3d(
        x=[positions[-1, 0]], y=[positions[-1, 1]], z=[positions[-1, 2]],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='End'
    ))

    fig.update_layout(
        title="KITTI Trajectory - 3D View",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)"
        ),
        height=600
    )

    return fig


def plot_speed_profile(speeds: np.ndarray, timestamps: np.ndarray):
    """Plot speed profile over time"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps[1:],  # Skip first frame
        y=speeds,
        mode='lines',
        name='Speed',
        line=dict(color='blue', width=2)
    ))

    fig.add_hline(y=np.mean(speeds), line_dash="dash",
                  annotation_text=f"Mean: {np.mean(speeds):.2f} m/s")

    fig.update_layout(
        title="Speed Profile Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Speed (m/s)",
        height=400
    )

    return fig


def plot_image_quality_distribution(images_df: pd.DataFrame):
    """Plot distribution of image quality metrics"""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Brightness', 'Contrast', 'Blur Metric',
                        'Feature Count', 'Feature Quality', 'Translation Speed'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )

    metrics = ['brightness', 'contrast', 'blur_metric', 'num_features', 'feature_quality', 'translation_speed']
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

    for metric, (row, col) in zip(metrics, positions):
        fig.add_trace(
            go.Histogram(x=images_df[metric], nbinsx=30, name=metric, showlegend=False),
            row=row, col=col
        )

    fig.update_layout(height=600, title_text="Image Quality Metrics Distribution")
    return fig


def plot_challenging_scenarios(challenges: Dict, total_frames: int):
    """Plot challenging scenarios timeline"""
    fig = go.Figure()

    colors = ['red', 'orange', 'yellow', 'purple', 'pink', 'brown']
    y_positions = list(range(len(challenges)))

    for i, (challenge_type, frames) in enumerate(challenges.items()):
        if frames:
            fig.add_trace(go.Scatter(
                x=frames,
                y=[i] * len(frames),
                mode='markers',
                name=challenge_type.replace('_', ' ').title(),
                marker=dict(size=8, color=colors[i])
            ))

    fig.update_layout(
        title="Challenging Scenarios Timeline",
        xaxis_title="Frame Number",
        yaxis=dict(
            tickmode='array',
            tickvals=y_positions,
            ticktext=[name.replace('_', ' ').title() for name in challenges.keys()]
        ),
        height=400
    )

    return fig


def plot_correlation_matrix(images_df: pd.DataFrame):
    """Plot correlation matrix of image metrics"""
    # Select numeric columns for correlation
    numeric_cols = ['brightness', 'contrast', 'blur_metric', 'num_features',
                    'feature_quality', 'translation_speed', 'rotation_speed']

    corr_matrix = images_df[numeric_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10}
    ))

    fig.update_layout(
        title="Correlation Matrix - Image Quality Metrics",
        height=500
    )

    return fig


def main():
    st.title("📊 KITTI Dataset EDA for Visual SLAM")
    st.write("Comprehensive Exploratory Data Analysis of KITTI Dataset for Visual SLAM Applications")

    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = KITTIDataAnalyzer()
        st.session_state.data_loaded = False

    # Sidebar
    st.sidebar.header("Dataset Configuration")

    # Data loading section
    st.sidebar.subheader("Load Data")
    data_source = st.sidebar.radio(
        "Data Source",
        options=["Demo Data", "Upload KITTI Data"],
        help="Choose between demo data or upload real KITTI dataset"
    )

    if data_source == "Demo Data":
        sequence_id = st.sidebar.selectbox("Sequence", ["00", "01", "02", "05", "07"])
        num_frames = st.sidebar.slider("Number of Frames", 100, 2000, 1000)

        if st.sidebar.button("Generate Demo Data") or not st.session_state.data_loaded:
            with st.spinner("Generating demo data..."):
                st.session_state.current_data = st.session_state.analyzer.generate_sample_data(
                    sequence_id, num_frames
                )
                st.session_state.data_loaded = True
                st.sidebar.success("Demo data generated!")

    else:
        st.sidebar.info("Upload your KITTI dataset files")
        # File upload widgets would go here for real implementation
        st.sidebar.write("Feature coming soon...")

    if not st.session_state.data_loaded:
        st.warning("Please load or generate data to proceed with analysis.")
        return

    data = st.session_state.current_data

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Trajectory Analysis",
        "🖼️ Image Quality",
        "⚠️ Challenging Scenarios",
        "📊 Statistical Analysis",
        "🔍 Detailed Insights"
    ])

    # Perform analysis
    trajectory_analysis = st.session_state.analyzer.analyze_trajectory(data['poses'])
    image_quality = st.session_state.analyzer.analyze_image_quality(data['images'])
    challenges = st.session_state.analyzer.detect_challenging_scenarios(
        data['images'], trajectory_analysis
    )

    with tab1:
        st.header("Trajectory Analysis")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Distance", f"{trajectory_analysis['total_distance']:.1f} m")
        with col2:
            st.metric("Average Speed", f"{trajectory_analysis['avg_speed']:.2f} m/s")
        with col3:
            st.metric("Max Speed", f"{trajectory_analysis['max_speed']:.2f} m/s")
        with col4:
            st.metric("Total Turns", f"{trajectory_analysis['total_turns']:.1f} rad")

        # 3D trajectory plot
        st.subheader("3D Trajectory Visualization")
        fig_3d = plot_trajectory_3d(trajectory_analysis['positions'], trajectory_analysis['speeds'])
        st.plotly_chart(fig_3d, use_container_width=True)

        # Speed profile
        st.subheader("Speed Profile")
        timestamps = data['images']['timestamp'].values
        fig_speed = plot_speed_profile(trajectory_analysis['speeds'], timestamps)
        st.plotly_chart(fig_speed, use_container_width=True)

        # Motion characteristics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Speed Distribution")
            fig_speed_hist = px.histogram(
                x=trajectory_analysis['speeds'],
                nbins=30,
                title="Speed Distribution",
                labels={'x': 'Speed (m/s)', 'y': 'Frequency'}
            )
            st.plotly_chart(fig_speed_hist, use_container_width=True)

        with col2:
            st.subheader("Acceleration Distribution")
            fig_acc_hist = px.histogram(
                x=trajectory_analysis['accelerations'],
                nbins=30,
                title="Acceleration Distribution",
                labels={'x': 'Acceleration (m/s²)', 'y': 'Frequency'}
            )
            st.plotly_chart(fig_acc_hist, use_container_width=True)

    with tab2:
        st.header("Image Quality Analysis")

        # Quality metrics overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Brightness", f"{data['images']['brightness'].mean():.1f}")
            st.metric("Brightness Std", f"{data['images']['brightness'].std():.1f}")
        with col2:
            st.metric("Avg Contrast", f"{data['images']['contrast'].mean():.1f}")
            st.metric("Contrast Std", f"{data['images']['contrast'].std():.1f}")
        with col3:
            st.metric("Avg Features", f"{data['images']['num_features'].mean():.0f}")
            st.metric("Feature Quality", f"{data['images']['feature_quality'].mean():.3f}")

        # Image quality distributions
        st.subheader("Quality Metrics Distribution")
        fig_quality = plot_image_quality_distribution(data['images'])
        st.plotly_chart(fig_quality, use_container_width=True)

        # Correlation analysis
        st.subheader("Correlation Analysis")
        fig_corr = plot_correlation_matrix(data['images'])
        st.plotly_chart(fig_corr, use_container_width=True)

        # Time series of quality metrics
        st.subheader("Quality Metrics Over Time")
        fig_time = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Brightness & Contrast', 'Features & Blur'),
            shared_xaxes=True
        )

        fig_time.add_trace(
            go.Scatter(x=data['images']['timestamp'], y=data['images']['brightness'],
                       name='Brightness', line=dict(color='orange')),
            row=1, col=1
        )
        fig_time.add_trace(
            go.Scatter(x=data['images']['timestamp'], y=data['images']['contrast'],
                       name='Contrast', line=dict(color='blue')),
            row=1, col=1
        )
        fig_time.add_trace(
            go.Scatter(x=data['images']['timestamp'], y=data['images']['num_features'],
                       name='Features', line=dict(color='green')),
            row=2, col=1
        )
        fig_time.add_trace(
            go.Scatter(x=data['images']['timestamp'], y=data['images']['blur_metric'] * 100,
                       name='Blur×100', line=dict(color='red')),
            row=2, col=1
        )

        fig_time.update_layout(height=600, title_text="Quality Metrics Timeline")
        st.plotly_chart(fig_time, use_container_width=True)

    with tab3:
        st.header("Challenging Scenarios for VSLAM")

        # Challenge summary
        st.subheader("Challenge Summary")
        challenge_counts = {k: len(v) for k, v in challenges.items()}

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Low Light Frames", challenge_counts['low_light'])
            st.metric("Low Contrast Frames", challenge_counts['low_contrast'])
        with col2:
            st.metric("Motion Blur Frames", challenge_counts['motion_blur'])
            st.metric("Few Features Frames", challenge_counts['few_features'])
        with col3:
            st.metric("Fast Motion Frames", challenge_counts['fast_motion'])
            st.metric("Sharp Turn Frames", challenge_counts['sharp_turns'])

        # Challenge timeline
        st.subheader("Challenging Scenarios Timeline")
        fig_challenges = plot_challenging_scenarios(challenges, len(data['images']))
        st.plotly_chart(fig_challenges, use_container_width=True)

        # Challenge severity heatmap
        st.subheader("Challenge Severity Analysis")
        challenge_matrix = np.zeros((len(data['images']), len(challenges)))
        for i, (challenge_type, frames) in enumerate(challenges.items()):
            for frame in frames:
                if frame < len(data['images']):
                    challenge_matrix[frame, i] = 1

        # Sample every 10th frame for visualization
        sample_indices = range(0, len(data['images']), max(1, len(data['images']) // 100))
        sampled_matrix = challenge_matrix[sample_indices, :]

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=sampled_matrix.T,
            x=[f"Frame {i}" for i in sample_indices],
            y=list(challenges.keys()),
            colorscale='Reds',
            showscale=True
        ))
        fig_heatmap.update_layout(
            title="Challenge Occurrence Heatmap (Sampled)",
            height=400
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with tab4:
        st.header("Statistical Analysis")

        # Detailed statistics
        st.subheader("Image Quality Statistics")
        stats_df = pd.DataFrame({
            'Metric': ['Brightness', 'Contrast', 'Blur', 'Features', 'Feature Quality'],
            'Mean': [
                data['images']['brightness'].mean(),
                data['images']['contrast'].mean(),
                data['images']['blur_metric'].mean(),
                data['images']['num_features'].mean(),
                data['images']['feature_quality'].mean()
            ],
            'Std': [
                data['images']['brightness'].std(),
                data['images']['contrast'].std(),
                data['images']['blur_metric'].std(),
                data['images']['num_features'].std(),
                data['images']['feature_quality'].std()
            ],
            'Min': [
                data['images']['brightness'].min(),
                data['images']['contrast'].min(),
                data['images']['blur_metric'].min(),
                data['images']['num_features'].min(),
                data['images']['feature_quality'].min()
            ],
            'Max': [
                data['images']['brightness'].max(),
                data['images']['contrast'].max(),
                data['images']['blur_metric'].max(),
                data['images']['num_features'].max(),
                data['images']['feature_quality'].max()
            ]
        })
        st.dataframe(stats_df, use_container_width=True)

        # Motion statistics
        st.subheader("Motion Statistics")
        motion_stats = pd.DataFrame({
            'Metric': ['Translation Speed', 'Rotation Speed', 'Acceleration'],
            'Mean': [
                trajectory_analysis['avg_speed'],
                data['images']['rotation_speed'].mean(),
                trajectory_analysis['avg_acceleration']
            ],
            'Max': [
                trajectory_analysis['max_speed'],
                data['images']['rotation_speed'].max(),
                trajectory_analysis['max_acceleration']
            ],
            'Std': [
                np.std(trajectory_analysis['speeds']),
                data['images']['rotation_speed'].std(),
                np.std(trajectory_analysis['accelerations'])
            ]
        })
        st.dataframe(motion_stats, use_container_width=True)

        # Box plots
        st.subheader("Quality Metrics Box Plots")
        fig_box = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Brightness', 'Contrast', 'Features')
        )

        fig_box.add_trace(go.Box(y=data['images']['brightness'], name='Brightness'), row=1, col=1)
        fig_box.add_trace(go.Box(y=data['images']['contrast'], name='Contrast'), row=1, col=2)
        fig_box.add_trace(go.Box(y=data['images']['num_features'], name='Features'), row=1, col=3)

        fig_box.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with tab5:
        st.header("Detailed Insights & Recommendations")

        # VSLAM suitability analysis
        st.subheader("VSLAM Suitability Analysis")

        # Calculate suitability scores
        brightness_score = 1 - (np.abs(data['images']['brightness'] - 128) / 128).mean()
        contrast_score = (data['images']['contrast'] / 100).mean()
        feature_score = min(1, data['images']['num_features'].mean() / 500)
        motion_score = 1 - min(1, trajectory_analysis['avg_speed'] / 20)  # Penalize very fast motion

        overall_score = (brightness_score + contrast_score + feature_score + motion_score) / 4

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Brightness Score", f"{brightness_score:.2f}")
        with col2:
            st.metric("Contrast Score", f"{contrast_score:.2f}")
        with col3:
            st.metric("Feature Score", f"{feature_score:.2f}")
        with col4:
            st.metric("Motion Score", f"{motion_score:.2f}")
        with col5:
            st.metric("Overall VSLAM Score", f"{overall_score:.2f}")

        # Recommendations
        st.subheader("Recommendations for VSLAM")

        recommendations = []

        if brightness_score < 0.7:
            recommendations.append("⚠️ **Lighting Issues**: Consider adaptive exposure or HDR techniques")

        if contrast_score < 0.5:
            recommendations.append("⚠️ **Low Contrast**: Apply histogram equalization or CLAHE preprocessing")

        if feature_score < 0.6:
            recommendations.append("⚠️ **Feature Detection**: Use more robust feature detectors like SIFT or SURF")

        if motion_score < 0.6:
            recommendations.append("⚠️ **Fast Motion**: Implement motion blur compensation or higher frame rate")

        if challenge_counts['low_light'] > len(data['images']) * 0.1:
            recommendations.append("🌙 **Low Light Adaptation**: Consider night-time SLAM techniques")

        if challenge_counts['sharp_turns'] > len(data['images']) * 0.05:
            recommendations.append("🔄 **Sharp Turns**: Implement gyroscope fusion for better tracking")

        if overall_score > 0.8:
            recommendations.append("✅ **Excellent**: This sequence is well-suited for VSLAM")
        elif overall_score > 0.6:
            recommendations.append("✅ **Good**: Minor optimizations recommended")
        else:
            recommendations.append("❌ **Challenging**: Significant preprocessing required")

        for rec in recommendations:
            st.write(rec)

        # Feature distribution analysis
        st.subheader("Feature Distribution Analysis")

        # Analyze feature density patterns
        low_feature_threshold = np.percentile(data['images']['num_features'], 25)
        high_feature_threshold = np.percentile(data['images']['num_features'], 75)

        feature_categories = []
        for features in data['images']['num_features']:
            if features < low_feature_threshold:
                feature_categories.append('Low')
            elif features > high_feature_threshold:
                feature_categories.append('High')
            else:
                feature_categories.append('Medium')

        feature_dist = pd.Series(feature_categories).value_counts()

        fig_feature_pie = px.pie(
            values=feature_dist.values,
            names=feature_dist.index,
            title="Feature Density Distribution"
        )
        st.plotly_chart(fig_feature_pie, use_container_width=True)

        # Motion pattern analysis
        st.subheader("Motion Pattern Analysis")

        # Classify motion types
        speed_threshold_low = np.percentile(trajectory_analysis['speeds'], 33)
        speed_threshold_high = np.percentile(trajectory_analysis['speeds'], 67)
        turn_threshold = np.percentile(trajectory_analysis['heading_changes'], 75)

        motion_patterns = []
        for i, (speed, turn) in enumerate(zip(trajectory_analysis['speeds'],
                                              trajectory_analysis['heading_changes'])):
            if speed < speed_threshold_low:
                if turn > turn_threshold:
                    motion_patterns.append('Slow Turning')
                else:
                    motion_patterns.append('Slow Straight')
            elif speed > speed_threshold_high:
                if turn > turn_threshold:
                    motion_patterns.append('Fast Turning')
                else:
                    motion_patterns.append('Fast Straight')
            else:
                if turn > turn_threshold:
                    motion_patterns.append('Medium Turning')
                else:
                    motion_patterns.append('Medium Straight')

        motion_dist = pd.Series(motion_patterns).value_counts()

        fig_motion = px.bar(
            x=motion_dist.index,
            y=motion_dist.values,
            title="Motion Pattern Distribution",
            labels={'x': 'Motion Pattern', 'y': 'Frame Count'}
        )
        fig_motion.update_xaxes(tickangle=45)
        st.plotly_chart(fig_motion, use_container_width=True)

        # Environmental conditions analysis
        st.subheader("Environmental Conditions Impact")

        # Create environmental condition categories based on image quality
        conditions = []
        for _, row in data['images'].iterrows():
            if row['brightness'] < 80:
                lighting = 'Dark'
            elif row['brightness'] > 180:
                lighting = 'Bright'
            else:
                lighting = 'Normal'

            if row['blur_metric'] > 3:
                blur_level = 'High'
            elif row['blur_metric'] > 1.5:
                blur_level = 'Medium'
            else:
                blur_level = 'Low'

            conditions.append(f"{lighting} Light, {blur_level} Blur")

        condition_dist = pd.Series(conditions).value_counts()

        fig_conditions = px.bar(
            x=condition_dist.values,
            y=condition_dist.index,
            orientation='h',
            title="Environmental Conditions Distribution",
            labels={'x': 'Frame Count', 'y': 'Condition'}
        )
        st.plotly_chart(fig_conditions, use_container_width=True)

        # SLAM performance prediction
        st.subheader("SLAM Performance Prediction")

        # Calculate frame-by-frame SLAM difficulty scores
        slam_difficulty = []
        for _, row in data['images'].iterrows():
            # Normalize metrics (0 = easy, 1 = difficult)
            brightness_difficulty = abs(row['brightness'] - 128) / 128
            contrast_difficulty = max(0, (50 - row['contrast']) / 50)
            blur_difficulty = min(1, row['blur_metric'] / 5)
            feature_difficulty = max(0, (300 - row['num_features']) / 300)
            motion_difficulty = min(1, row['translation_speed'] / 10)

            overall_difficulty = (brightness_difficulty + contrast_difficulty +
                                  blur_difficulty + feature_difficulty + motion_difficulty) / 5
            slam_difficulty.append(overall_difficulty)

        # Plot difficulty over time
        fig_difficulty = go.Figure()
        fig_difficulty.add_trace(go.Scatter(
            x=data['images']['timestamp'],
            y=slam_difficulty,
            mode='lines',
            name='SLAM Difficulty',
            line=dict(color='red', width=2)
        ))

        # Add difficulty thresholds
        fig_difficulty.add_hline(y=0.3, line_dash="dash", line_color="green",
                                 annotation_text="Easy Threshold")
        fig_difficulty.add_hline(y=0.7, line_dash="dash", line_color="orange",
                                 annotation_text="Difficult Threshold")

        fig_difficulty.update_layout(
            title="Predicted SLAM Difficulty Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Difficulty Score (0=Easy, 1=Difficult)",
            height=400
        )
        st.plotly_chart(fig_difficulty, use_container_width=True)

        # Summary statistics
        easy_frames = sum(1 for d in slam_difficulty if d < 0.3)
        medium_frames = sum(1 for d in slam_difficulty if 0.3 <= d < 0.7)
        difficult_frames = sum(1 for d in slam_difficulty if d >= 0.7)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Easy Frames", f"{easy_frames} ({easy_frames / len(slam_difficulty) * 100:.1f}%)")
        with col2:
            st.metric("Medium Frames", f"{medium_frames} ({medium_frames / len(slam_difficulty) * 100:.1f}%)")
        with col3:
            st.metric("Difficult Frames", f"{difficult_frames} ({difficult_frames / len(slam_difficulty) * 100:.1f}%)")

    # Export section
    st.sidebar.subheader("Export Results")
    if st.sidebar.button("Export Analysis Report"):
        # Create comprehensive report
        report = {
            'sequence_id': data['sequence_id'],
            'total_frames': len(data['images']),
            'trajectory_analysis': trajectory_analysis,
            'image_quality_stats': {
                'brightness': data['images']['brightness'].describe().to_dict(),
                'contrast': data['images']['contrast'].describe().to_dict(),
                'features': data['images']['num_features'].describe().to_dict()
            },
            'challenge_counts': challenge_counts,
            'slam_suitability_scores': {
                'brightness_score': brightness_score,
                'contrast_score': contrast_score,
                'feature_score': feature_score,
                'motion_score': motion_score,
                'overall_score': overall_score
            },
            'recommendations': recommendations
        }

        # Convert to JSON for download
        import json
        report_json = json.dumps(report, indent=2, default=str)

        st.sidebar.download_button(
            label="Download Report (JSON)",
            data=report_json,
            file_name=f"kitti_eda_report_{data['sequence_id']}.json",
            mime="application/json"
        )

        # Also create CSV summary
        summary_df = pd.DataFrame({
            'Metric': ['Total Distance (m)', 'Avg Speed (m/s)', 'Max Speed (m/s)',
                       'Avg Brightness', 'Avg Contrast', 'Avg Features',
                       'Low Light Frames', 'Fast Motion Frames', 'Overall SLAM Score'],
            'Value': [
                f"{trajectory_analysis['total_distance']:.2f}",
                f"{trajectory_analysis['avg_speed']:.2f}",
                f"{trajectory_analysis['max_speed']:.2f}",
                f"{data['images']['brightness'].mean():.1f}",
                f"{data['images']['contrast'].mean():.1f}",
                f"{data['images']['num_features'].mean():.0f}",
                str(challenge_counts['low_light']),
                str(challenge_counts['fast_motion']),
                f"{overall_score:.3f}"
            ]
        })

        csv_data = summary_df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download Summary (CSV)",
            data=csv_data,
            file_name=f"kitti_summary_{data['sequence_id']}.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()