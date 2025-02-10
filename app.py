import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ------------------------------------------------------------------------------
# Page Configuration and App Title
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Smile Detection 😄", page_icon="😃", layout="wide")
st.title("Smile Detection with OpenCV and Streamlit 😄")
st.write("Welcome! This app detects your smile in real-time. Keep smiling and enjoy! 😊")

# ------------------------------------------------------------------------------
# Utility: Load Haar Cascade Models with Caching
# ------------------------------------------------------------------------------
@st.cache_resource
def load_cascades():
    """
    Load Haar Cascade models for face and smile detection.
    Returns:
        face_cascade (cv2.CascadeClassifier): Haar cascade for face detection.
        smile_cascade (cv2.CascadeClassifier): Haar cascade for smile detection.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
    
    if face_cascade.empty():
        st.error("Failed to load face cascade. Please check the file path for 'haarcascade_frontalface_default.xml'.")
    if smile_cascade.empty():
        st.error("Failed to load smile cascade. Please check the file path for 'haarcascade_smile.xml'.")
    
    return face_cascade, smile_cascade

face_cascade, smile_cascade = load_cascades()

# ------------------------------------------------------------------------------
# Core Function: Detect Faces and Smiles
# ------------------------------------------------------------------------------
def detect_smiles(frame, scale=0.5):
    """
    Detect faces and smiles in the provided frame and annotate them.

    Args:
        frame (np.ndarray): The input image frame (in BGR format).
        scale (float): Downscaling factor to speed up processing.

    Returns:
        np.ndarray: The annotated image frame.
    """
    # Downscale the frame for faster processing.
    small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image.
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Process each detected face.
    for (x, y, w, h) in faces:
        # Scale the coordinates back to the original frame size.
        x_orig, y_orig, w_orig, h_orig = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
        cv2.rectangle(frame, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), (255, 0, 0), 2)

        # Define the region of interest (ROI) for the face.
        roi_gray = gray[y:y + h, x:x + w]

        # Detect smiles within the ROI. Adjust parameters as needed.
        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.7,
            minNeighbors=15,
            minSize=(20, 20)
        )

        # For each smile detected, annotate the original frame.
        for (sx, sy, sw, sh) in smiles:
            sx_orig, sy_orig, sw_orig, sh_orig = int(sx / scale), int(sy / scale), int(sw / scale), int(sh / scale)
            cv2.rectangle(
                frame,
                (x_orig + sx_orig, y_orig + sy_orig),
                (x_orig + sx_orig + sw_orig, y_orig + sy_orig + sh_orig),
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                "Smile Detected! 😄",
                (x_orig, y_orig - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
    return frame

# ------------------------------------------------------------------------------
# Video Transformer for Real-Time Smile Detection using streamlit-webrtc
# ------------------------------------------------------------------------------
class SmileDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        """
        Process each video frame to detect faces and smiles.
        The input frame is provided in an 'av.VideoFrame' format.
        Returns:
            The annotated frame in BGR format.
        """
        # Convert the incoming frame to a numpy array (BGR format).
        img = frame.to_ndarray(format="bgr24")

        # Mirror the frame for a natural selfie view.
        img = cv2.flip(img, 1)

        # Process the frame to detect and annotate smiles.
        img = detect_smiles(img, scale=0.5)
        return img

# ------------------------------------------------------------------------------
# Run the Real-Time Smile Detection via WebRTC
# ------------------------------------------------------------------------------
st.write("Please allow access to your webcam for real-time smile detection.")
webrtc_streamer(key="smile-detection", video_transformer_factory=SmileDetectionTransformer)
