import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import pickle
import os
from typing import List, Tuple, Dict, Optional
import threading
import time

# Set page config
st.set_page_config(
    page_title="Visual SLAM System",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)


class VisualSLAM:
    """Visual SLAM implementation for KITTI dataset"""

    def __init__(self):
        # Camera intrinsic parameters (KITTI dataset - left camera)
        self.K = np.array([
            [7.215377e+02, 0.000000e+00, 6.095593e+02],
            [0.000000e+00, 7.215377e+02, 1.728540e+02],
            [0.000000e+00, 0.000000e+00, 1.000000e+00]
        ])

        # Feature detector and matcher
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # SLAM state
        self.poses = []  # Camera poses
        self.map_points = []  # 3D map points
        self.keyframes = []  # Keyframe images
        self.trajectory = []  # 2D trajectory for visualization

        # Parameters
        self.min_matches = 50
        self.keyframe_threshold = 30
        self.max_reprojection_error = 1.0

    def extract_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract ORB features from image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if descriptors is None:
            return np.array([]), np.array([])

        # Convert keypoints to numpy array
        points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        return points, descriptors

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        """Match features between two frames"""
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []

        matches = self.matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Filter matches by distance
        good_matches = []
        if len(matches) > 0:
            max_dist = matches[0].distance * 2.5
            good_matches = [m for m in matches if m.distance < max_dist]

        return good_matches[:200]  # Limit number of matches

    def estimate_pose(self, points1: np.ndarray, points2: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Estimate camera pose between two frames"""
        if len(points1) < 8 or len(points2) < 8:
            return None

        # Find essential matrix
        E, mask = cv2.findEssentialMat(
            points1, points2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:
            return None

        # Recover pose from essential matrix
        _, R, t, _ = cv2.recoverPose(E, points1, points2, self.K)

        return R, t

    def triangulate_points(self, points1: np.ndarray, points2: np.ndarray,
                           R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Triangulate 3D points from stereo correspondence"""
        # Create projection matrices
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t])

        # Triangulate points
        points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
        points_3d = points_4d[:3] / points_4d[3]

        return points_3d.T

    def process_frame(self, image: np.ndarray, frame_id: int) -> Dict:
        """Process a single frame for SLAM"""
        # Extract features
        points, descriptors = self.extract_features(image)

        result = {
            'frame_id': frame_id,
            'features': len(points),
            'pose': None,
            'new_points': 0
        }

        if len(self.keyframes) == 0:
            # First frame - initialize
            pose = np.eye(4)
            self.poses.append(pose)
            self.keyframes.append({
                'image': image,
                'points': points,
                'descriptors': descriptors,
                'pose': pose
            })
            self.trajectory.append([0, 0])
            result['pose'] = pose

        else:
            # Match with previous keyframe
            prev_keyframe = self.keyframes[-1]
            matches = self.match_features(prev_keyframe['descriptors'], descriptors)

            if len(matches) < self.min_matches:
                return result

            # Get matched points
            prev_points = np.array([prev_keyframe['points'][m.queryIdx] for m in matches])
            curr_points = np.array([points[m.trainIdx] for m in matches])

            # Estimate pose
            pose_result = self.estimate_pose(prev_points, curr_points)
            if pose_result is None:
                return result

            R_rel, t_rel = pose_result

            # Compute absolute pose
            prev_pose = self.poses[-1]
            R_abs = prev_pose[:3, :3] @ R_rel
            t_abs = prev_pose[:3, :3] @ t_rel + prev_pose[:3, 3:4]

            current_pose = np.eye(4)
            current_pose[:3, :3] = R_abs
            current_pose[:3, 3:4] = t_abs

            self.poses.append(current_pose)

            # Add to trajectory
            self.trajectory.append([t_abs[0, 0], t_abs[2, 0]])  # x, z coordinates

            # Triangulate new 3D points
            if len(matches) > 20:
                points_3d = self.triangulate_points(prev_points, curr_points, R_rel, t_rel)

                # Filter points by depth
                valid_points = points_3d[points_3d[:, 2] > 0]
                valid_points = valid_points[valid_points[:, 2] < 50]  # Max depth 50m

                if len(valid_points) > 0:
                    self.map_points.extend(valid_points)
                    result['new_points'] = len(valid_points)

            # Check if we need a new keyframe
            if len(matches) < self.keyframe_threshold or frame_id % 10 == 0:
                self.keyframes.append({
                    'image': image,
                    'points': points,
                    'descriptors': descriptors,
                    'pose': current_pose
                })

            result['pose'] = current_pose
            result['matches'] = len(matches)

        return result


def create_kitti_demo_data():
    """Create demo data simulating KITTI dataset"""
    np.random.seed(42)

    # Generate synthetic trajectory (figure-8 pattern)
    t = np.linspace(0, 4 * np.pi, 100)
    x = 10 * np.sin(t)
    z = 5 * np.sin(2 * t)

    demo_data = []
    for i in range(len(t)):
        # Create synthetic image with features
        img = np.random.randint(0, 255, (376, 1241, 3), dtype=np.uint8)

        # Add some feature-like patterns
        for _ in range(50):
            center = (np.random.randint(50, 1191), np.random.randint(50, 326))
            cv2.circle(img, center, np.random.randint(3, 8),
                       (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255)), -1)

        demo_data.append({
            'image': img,
            'true_pose': [x[i], 0, z[i]]
        })

    return demo_data


def plot_trajectory_3d(poses: List[np.ndarray], map_points: List[np.ndarray]):
    """Create 3D trajectory plot using Plotly"""
    if not poses:
        return go.Figure()

    # Extract positions
    positions = np.array([[pose[0, 3], pose[1, 3], pose[2, 3]] for pose in poses])

    fig = go.Figure()

    # Add trajectory
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='lines+markers',
        name='Camera Trajectory',
        line=dict(color='red', width=4),
        marker=dict(size=4, color='red')
    ))

    # Add map points
    if map_points and len(map_points) > 0:
        map_array = np.array(map_points)
        if len(map_array) > 1000:  # Subsample for performance
            indices = np.random.choice(len(map_array), 1000, replace=False)
            map_array = map_array[indices]

        fig.add_trace(go.Scatter3d(
            x=map_array[:, 0],
            y=map_array[:, 1],
            z=map_array[:, 2],
            mode='markers',
            name='Map Points',
            marker=dict(size=2, color='blue', opacity=0.6)
        ))

    fig.update_layout(
        title="Visual SLAM - 3D Trajectory and Map",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=0.6)
            )
        ),
        height=500
    )

    return fig


def plot_2d_trajectory(trajectory: List[List[float]]):
    """Create 2D trajectory plot"""
    if not trajectory:
        return go.Figure()

    traj_array = np.array(trajectory)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=traj_array[:, 0],
        y=traj_array[:, 1],
        mode='lines+markers',
        name='Trajectory (Top View)',
        line=dict(color='red', width=3),
        marker=dict(size=6, color='red')
    ))

    fig.update_layout(
        title="Camera Trajectory (Top View)",
        xaxis_title="X (m)",
        yaxis_title="Z (m)",
        height=400,
        showlegend=False
    )

    return fig


def main():
    st.title("🗺️ Visual SLAM System")
    st.write("Real-time Visual Simultaneous Localization and Mapping using KITTI dataset")

    # Sidebar controls
    st.sidebar.header("SLAM Controls")

    # Initialize SLAM system
    if 'slam' not in st.session_state:
        st.session_state.slam = VisualSLAM()
        st.session_state.demo_data = create_kitti_demo_data()
        st.session_state.current_frame = 0
        st.session_state.is_running = False

    # Control buttons
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("▶️ Start"):
            st.session_state.is_running = True
    with col2:
        if st.button("⏸️ Pause"):
            st.session_state.is_running = False
    with col3:
        if st.button("🔄 Reset"):
            st.session_state.slam = VisualSLAM()
            st.session_state.current_frame = 0
            st.session_state.is_running = False

    # Parameters
    st.sidebar.subheader("Parameters")
    min_matches = st.sidebar.slider("Min Matches", 10, 100, 50)
    keyframe_threshold = st.sidebar.slider("Keyframe Threshold", 10, 50, 30)

    st.session_state.slam.min_matches = min_matches
    st.session_state.slam.keyframe_threshold = keyframe_threshold

    # Main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Current Frame")
        frame_placeholder = st.empty()

        st.subheader("3D Trajectory and Map")
        plot_3d_placeholder = st.empty()

    with col2:
        st.subheader("SLAM Statistics")
        stats_placeholder = st.empty()

        st.subheader("2D Trajectory")
        plot_2d_placeholder = st.empty()

    # Auto-advance frames
    if st.session_state.is_running and st.session_state.current_frame < len(st.session_state.demo_data):
        # Process current frame
        current_data = st.session_state.demo_data[st.session_state.current_frame]
        result = st.session_state.slam.process_frame(
            current_data['image'],
            st.session_state.current_frame
        )

        # Update display
        with frame_placeholder.container():
            st.image(current_data['image'], caption=f"Frame {st.session_state.current_frame}", width=600)

        # Update statistics
        with stats_placeholder.container():
            st.metric("Frame", st.session_state.current_frame)
            st.metric("Features Detected", result.get('features', 0))
            st.metric("Feature Matches", result.get('matches', 0))
            st.metric("New 3D Points", result.get('new_points', 0))
            st.metric("Total Keyframes", len(st.session_state.slam.keyframes))
            st.metric("Total Map Points", len(st.session_state.slam.map_points))

            if result['pose'] is not None:
                pos = result['pose'][:3, 3]
                st.write(f"**Position:** ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

        # Update plots
        with plot_3d_placeholder.container():
            fig_3d = plot_trajectory_3d(st.session_state.slam.poses, st.session_state.slam.map_points)
            st.plotly_chart(fig_3d, use_container_width=True)

        with plot_2d_placeholder.container():
            fig_2d = plot_2d_trajectory(st.session_state.slam.trajectory)
            st.plotly_chart(fig_2d, use_container_width=True)

        # Advance frame
        st.session_state.current_frame += 1

        # Auto-refresh
        time.sleep(0.1)
        st.rerun()

    elif st.session_state.current_frame >= len(st.session_state.demo_data):
        st.success("🎉 SLAM processing completed!")
        st.session_state.is_running = False

    # Manual frame control
    st.sidebar.subheader("Manual Control")
    manual_frame = st.sidebar.slider(
        "Frame", 0, len(st.session_state.demo_data) - 1,
        st.session_state.current_frame
    )

    if manual_frame != st.session_state.current_frame:
        st.session_state.current_frame = manual_frame
        st.rerun()


if __name__ == "__main__":
    main()