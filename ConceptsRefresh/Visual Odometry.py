import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set page title
st.title("Monocular Visual Odometry")

# Initialize session state
if "running" not in st.session_state:
    st.session_state.running = False
if "trajectory" not in st.session_state:
    st.session_state.trajectory = [[0, 0]]  # [x, z] coordinates

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    st.session_state.running = st.toggle("Run Visual Odometry", value=False)
    focal_length = st.slider("Focal Length (px)", 100, 2000, 700)
    cx, cy = st.slider("Principal Point (cx, cy)", 0, 640, (320, 240), step=10)

# Initialize camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    st.error("Cannot open webcam")
    st.stop()

# Camera intrinsic matrix
K = np.array([[focal_length, 0, cx],
              [0, focal_length, cy],
              [0, 0, 1]], dtype=np.float32)

# Initialize ORB and matcher
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Placeholders for video and trajectory plot
video_placeholder = st.empty()
plot_placeholder = st.empty()

# Initialize variables
prev_kp, prev_des = None, None
prev_frame = None
current_pose = np.eye(4)  # 4x4 transformation matrix
trajectory = st.session_state.trajectory
points3D = None  # Global 3D points

try:
    while st.session_state.running:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture frame")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ORB keypoints
        kp, des = orb.detectAndCompute(gray, None)
        frame_with_kp = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0),
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Display video feed
        video_placeholder.image(frame_with_kp, channels="BGR", caption="Live Feed with ORB Keypoints")

        # Process if previous frame exists
        if prev_kp is not None and prev_des is not None and des is not None:
            # Match features
            matches = bf.match(prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            # Extract matched points
            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

            if len(matches) > 8:
                # Estimate essential matrix
                E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None and mask is not None:
                    # Filter points with mask
                    valid_indices = mask.ravel() == 1
                    if np.sum(valid_indices) > 8:  # Ensure enough inliers
                        pts1_valid = pts1[valid_indices]
                        pts2_valid = pts2[valid_indices]

                        # Recover relative pose
                        _, R, t, _ = cv2.recoverPose(E, pts1_valid, pts2_valid, K)

                        # Triangulate points for initial 3D points
                        if points3D is None:
                            P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
                            P2 = np.hstack((R, t))
                            points4D = cv2.triangulatePoints(K @ P1, K @ P2, pts1_valid.T, pts2_valid.T)
                            points3D = points4D[:3] / points4D[3]
                            points3D = points3D.T
                        else:
                            # Use PnP to estimate pose
                            if len(points3D) >= np.sum(valid_indices):
                                # Ensure points3D and pts2_valid have matching sizes
                                valid_points3D = points3D[:np.sum(valid_indices)]
                                _, rvec, tvec, inliers = cv2.solvePnPRansac(
                                    valid_points3D, pts2_valid, K, None, iterationsCount=100, reprojectionError=8.0
                                )
                                if inliers is not None:
                                    R, _ = cv2.Rodrigues(rvec)
                                    t = tvec

                                    # Update pose
                                    T = np.eye(4)
                                    T[:3, :3] = R
                                    T[:3, 3] = t.ravel()
                                    current_pose = current_pose @ np.linalg.inv(T)

                                    # Add to trajectory
                                    trajectory.append([current_pose[0, 3], current_pose[2, 3]])

                        # Update 3D points for next iteration
                        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
                        P2 = np.hstack((R, t))
                        points4D = cv2.triangulatePoints(K @ P1, K @ P2, pts1_valid.T, pts2_valid.T)
                        points3D = points4D[:3] / points4D[3]
                        points3D = points3D.T

                        # Plot trajectory
                        fig, ax = plt.subplots()
                        traj_x, traj_z = zip(*trajectory)
                        ax.plot(traj_x, traj_z, 'b-', label="Estimated Trajectory")
                        ax.set_xlabel("X (m)")
                        ax.set_ylabel("Z (m)")
                        ax.legend()
                        ax.grid(True)
                        plot_placeholder.pyplot(fig)

        # Update previous frame data
        prev_kp, prev_des = kp, des
        prev_frame = gray

        # Small delay to stabilize processing
        cv2.waitKey(500)

except Exception as e:
    st.error(f"Error: {str(e)}")
finally:
    camera.release()

# Save trajectory to session state
st.session_state.trajectory = trajectory