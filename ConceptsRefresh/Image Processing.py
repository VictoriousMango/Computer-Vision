import streamlit as st
import cv2

st.title("Live Camera Feed")
if "sidebar_Header" not in st.session_state:
    st.session_state["sidebar_Header"] = "Original Video"
stframe= st.empty()
camera = cv2.VideoCapture(0)
# tab1, tab2, tab3 = st.tabs(["Normal Video Streaming", "Canny Edge Detection", "Harris Corner Detection"])
with st.sidebar:
    st.sidebar.header(st.session_state["sidebar_Header"])
    run = st.toggle("Stop", value=False, disabled=False, label_visibility="visible")
    display = st.radio(
        "Select the Display",
        ["Original Video", "Canny Edge Detection", "Harris Corner Detection", "SIFT", "SURF"]
    )
    if display == "Original Video":
        st.session_state["sidebar_Header"]="Original Video"
    elif display == "Canny Edge Detection":
        st.session_state["sidebar_Header"]="Parameters for Canny Edge Detector"
        thres_l, thres_u = st.slider("Limits", 0, 255, (50, 75))
    elif display == "Harris Corner Detection":
        st.session_state["sidebar_Header"]="Parameters of Harris Corner Detection"
        block_size = st.slider("Block Size", 1, 10, 2)
        ksize = st.slider("Sobel Kernel Size", 1, 7, 3, step=2)
        k = st.slider("Harris Parameter k", 0.01, 0.1, 0.04, step=0.01)
    elif display == "SIFT":
        st.session_state["sidebar_Header"] = "Parameters of SIFT"
        nfeatures = st.slider("Max Feature", 0, 1000, 500)
        sift = cv2.SIFT_create(nfeatures=nfeatures)
    elif display == "SURF":
        st.session_state["sidebar_Header"] = "Parameter of SURF"
        hessian_threshold = st.slider("Hessian Threshold", 100, 1000, 400)
        surf = cv2.xfeatures2d.SURF_create(hessian_threshold=hessian_threshold)
if not camera.isOpened():
    st.warning("Cannot Open Webcam")
else:
    while not run:
        ret, frame = camera.read()
        # display, channels = frame, "BGR"
        if not ret:
            st.warning("Failed to Capture frame")
            break
        if display == "Original Video":
            stframe.image(frame, channels="BGR")
        elif display == "Canny Edge Detection":
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edge = cv2.Canny(grey, thres_l, thres_u)
            stframe.image(edge, channels="GRAY")
        elif display == "Harris Corner Detection":
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = cv2.cornerHarris(grey, block_size, ksize, k)
            corners = cv2.dilate(corners, None)
            frame[corners > 0.01*corners.max()] = [0, 0, 255]
            stframe.image(frame, channels="BGR")
        elif display == "SIFT":
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = sift.detectAndCompute(grey, None)
            frame_with_kp = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            stframe.image(frame_with_kp, channels="BGR")
        elif display == "SURF":
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = surf.detectAndCompute(grey, None)
            frame_with_kp = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            stframe.image(frame_with_kp, channels="BGR")

    cv2.waitKey(10)
    camera.release()