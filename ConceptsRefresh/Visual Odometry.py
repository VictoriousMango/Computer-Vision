import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

st.title("Visual Odometry with Kitti Dataset")

# Kitti Dataset Directory Path inclusion
sequence = st.selectbox("Select KITTI Sequence", [f"{i:02d}" for i in range(11)])
data_path=st.text_input("Dataset Path", "./dataset")
seq_path = os.path.join(data_path, "sequences", sequence)
calib_file = os.path.join(seq_path, "calib.txt")
pose_file = os.path.join(data_path, "poses", f"{sequence}.txt")

# Read Callibration
calib = pd.read_csv(calib_file, delimiter=" ", header=None, index_col=0)
P0 = np.array(calib.loc["P0"]).reshape(3, 4)
K = P0[: , :3]
basline = 0.54 # Kitti Baseline in meters
focal_length = k[0,0]

# Read the ground truth poses
poses = pd.read_csv(pose_file, delimiter=" ", header=None)
gt_poses = [np.array(poses.iloc[i]).reshape(3, 4) for i in range(len(poses))]

# Left Images

# Right Images
