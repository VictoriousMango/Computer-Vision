import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import zipfile
import tempfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------------
# Utility Functions
# ------------------------
def load_calibration(calib_file):
    lines = calib_file.read().decode('utf-8').splitlines()
    first_line = lines[0].split()
    if len(first_line) == 12:
        P0 = np.array([float(x) for x in first_line]).reshape(3, 4)
        return P0[:, :3]  # Intrinsic matrix
    else:
        raise ValueError("Calibration file does not contain 12 elements per line.")

def load_poses(poses_file):
    lines = poses_file.read().decode('utf-8').splitlines()
    poses = [np.array([float(x) for x in line.split()]).reshape(3, 4) for line in lines if len(line.split()) == 12]
    return poses

def get_image_pairs(image_dir):
    files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')])
    image_paths = [os.path.join(image_dir, f) for f in files]
    return list(zip(image_paths[:-1], image_paths[1:]))

def process_frame_pair(img1_path, img2_path, K):
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 8:
        return None, None, [], kp1, kp2
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, good, kp1, kp2

def draw_trajectory(predicted_poses, gt_poses=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pred_coords = np.array([pose[:, 3] for pose in predicted_poses])
    ax.plot(pred_coords[:, 0], pred_coords[:, 1], pred_coords[:, 2], label='Predicted Trajectory', marker='o')

    if gt_poses:
        gt_coords = np.array([pose[:, 3] for pose in gt_poses[:len(predicted_poses)]])
        ax.plot(gt_coords[:, 0], gt_coords[:, 1], gt_coords[:, 2], label='Ground Truth Trajectory', marker='x')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    st.pyplot(fig)

# ------------------------
# Streamlit App
# ------------------------
st.set_page_config(page_title="VSLAM Full App", layout="wide")
st.title("Visual SLAM with KITTI Dataset or Live Camera")

option = st.sidebar.radio("Choose Mode", ["Live Camera", "Upload KITTI Dataset"])

if option == "Upload KITTI Dataset":
    image_folder = st.sidebar.file_uploader("Upload Image Folder (.zip)", type='zip')
    calib_file = st.sidebar.file_uploader("Upload Calibration File (.txt)", type='txt')
    poses_file = st.sidebar.file_uploader("Upload Poses File (.txt)", type='txt')

    if image_folder and calib_file and poses_file:
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(image_folder, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        K = load_calibration(calib_file)
        poses_gt = load_poses(poses_file)
        image_pairs = get_image_pairs(temp_dir)
        st.success(f"Found {len(image_pairs)} image pairs")

        trajectory = []
        pose = np.eye(4)
        trajectory.append(pose[:3])

        max_frames_to_show = 50

        for i, (img1, img2) in enumerate(image_pairs):
            R, t, matches, kp1, kp2 = process_frame_pair(img1, img2, K)
            if R is None:
                continue
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3:] = t
            pose = pose @ np.linalg.inv(T)
            trajectory.append(pose[:3])

            if i < max_frames_to_show:
                img1_color = cv2.imread(img1)
                img2_color = cv2.imread(img2)
                match_img = cv2.drawMatches(img1_color, kp1, img2_color, kp2, matches[:30], None, flags=2)
                st.image(match_img, caption=f"Feature Matches Frame {i}", channels="BGR")

        st.subheader("Estimated vs Ground Truth 3D Trajectory")
        draw_trajectory(trajectory, poses_gt)

elif option == "Live Camera":
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    orb = cv2.ORB_create(1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    prev_gray, prev_kp, prev_des = None, None, None
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]])
    trajectory = []
    pose = np.eye(4)
    trajectory.append(pose[:3])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)

        if prev_gray is not None and prev_des is not None:
            matches = bf.match(prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3:] = t
                pose = pose @ np.linalg.inv(T)
                trajectory.append(pose[:3])

                match_img = cv2.drawMatches(prev_frame, prev_kp, frame, kp, matches[:30], None, flags=2)
                stframe.image(match_img, channels="BGR")

        prev_gray = gray
        prev_kp = kp
        prev_des = des
        prev_frame = frame

    cap.release()
    st.subheader("Live Camera Estimated Trajectory")
    draw_trajectory(trajectory)
    cv2.destroyAllWindows()

# ------------------------
# ROS2 Integration Placeholder
# ------------------------
# Future enhancement: use rclpy to create a node to publish pose estimates
# Example stub:
# import rclpy
# from geometry_msgs.msg import PoseStamped
# def ros2_node():
#     rclpy.init()
#     node = rclpy.create_node('vslam_publisher')
#     pub = node.create_publisher(PoseStamped, 'vslam_pose', 10)
#     ... publish pose from trajectory
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()
