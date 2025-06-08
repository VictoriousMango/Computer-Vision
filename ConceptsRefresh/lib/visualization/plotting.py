import numpy as np
import matplotlib.pyplot as plt

def visualize_paths(gt_path, pred_path, title="Visual Odometry", file_out=None):
    gt_path = np.array(gt_path)
    pred_path = np.array(pred_path)

    gt_x, gt_y = gt_path.T
    pred_x, pred_y = pred_path.T

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.plot(gt_x, gt_y, label='Ground Truth', color='blue', linewidth=2)
    plt.plot(pred_x, pred_y, label='Predicted Path', color='green', linestyle='--')
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    if file_out:
        plt.savefig(file_out)
    plt.show()

def plot_error(gt_path, pred_path):
    gt_path = np.array(gt_path)
    pred_path = np.array(pred_path)

    error = np.linalg.norm(gt_path - pred_path, axis=1)

    plt.figure(figsize=(10, 4))
    plt.title("Pose Estimation Error over Frames")
    plt.plot(error, color='red')
    plt.xlabel("Frame Index")
    plt.ylabel("Error Magnitude")
    plt.grid(True)
    plt.show()
