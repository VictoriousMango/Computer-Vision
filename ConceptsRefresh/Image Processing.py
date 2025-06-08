import streamlit as st
import cv2
import numpy as np

# Set page title
st.title("Live Camera Feed with Feature Detection")

# Initialize session state for sidebar header
if "sidebar_Header" not in st.session_state:
    st.session_state["sidebar_Header"] = "Original Video"

# Placeholder for video frame
stframe = st.empty()

# Initialize webcam
camera = cv2.VideoCapture(0)

# Sidebar for selecting display mode and parameters
with st.sidebar:
    st.header(st.session_state["sidebar_Header"])
    run = st.toggle("Stop", value=True, label_visibility="visible")
    display = st.radio(
        "Select the Display",
        ["Original Video", "Canny Edge Detection", "Harris Corner Detection", "SIFT", "ORB"],
        index=0
    )

    # Update sidebar header based on selection
    if display == "Original Video":
        st.session_state["sidebar_Header"] = "Original Video"
    elif display == "Canny Edge Detection":
        st.session_state["sidebar_Header"] = "Parameters for Canny Edge Detector"
        thres_l, thres_u = st.slider("Thresholds (Lower, Upper)", 0, 255, (50, 150))
    elif display == "Harris Corner Detection":
        st.session_state["sidebar_Header"] = "Parameters for Harris Corner Detection"
        block_size = st.slider("Block Size", 1, 10, 2)
        ksize = st.slider("Sobel Kernel Size", 1, 7, 3, step=2)
        k = st.slider("Harris Parameter k", 0.01, 0.1, 0.04, step=0.01)
    elif display == "SIFT":
        st.session_state["sidebar_Header"] = "Parameters for SIFT"
        nfeatures = st.slider("Max Features (SIFT)", 0, 1000, 500)
        sift = cv2.SIFT_create(nfeatures=nfeatures)
    elif display == "ORB":
        st.session_state["sidebar_Header"] = "Parameters for ORB"
        nfeatures = st.slider("Max Features (ORB)", 0, 1000, 500)
        orb = cv2.ORB_create(nfeatures=nfeatures)

# Check if webcam is opened successfully
if not camera.isOpened():
    st.error("Cannot open webcam")
else:
    try:
        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture frame")
                break

            # Process frame based on selected display
            if display == "Original Video":
                stframe.image(frame, channels="BGR", caption="Original Video")
            elif display == "Canny Edge Detection":
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edge = cv2.Canny(grey, thres_l, thres_u)
                stframe.image(edge, channels="GRAY", caption="Canny Edge Detection")
            elif display == "Harris Corner Detection":
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners = cv2.cornerHarris(grey, block_size, ksize, k)
                corners = cv2.dilate(corners, None)
                frame_with_corners = frame.copy()
                frame_with_corners[corners > 0.01 * corners.max()] = [0, 0, 255]
                stframe.image(frame_with_corners, channels="BGR", caption="Harris Corner Detection")
            elif display == "SIFT":
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                keypoints, descriptors = sift.detectAndCompute(grey, None)
                frame_with_kp = cv2.drawKeypoints(
                    frame, keypoints, None, color=(0, 255, 0),
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                )
                stframe.image(frame_with_kp, channels="BGR", caption="SIFT Keypoints")
            elif display == "ORB":
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                keypoints, descriptors = orb.detectAndCompute(grey, None)
                frame_with_kp = cv2.drawKeypoints(
                    frame, keypoints, None, color=(0, 255, 0),
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                )
                stframe.image(frame_with_kp, channels="BGR", caption="ORB Keypoints")

            # Small delay to prevent overwhelming the app
            cv2.waitKey(10)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        # Ensure camera is released
        camera.release()