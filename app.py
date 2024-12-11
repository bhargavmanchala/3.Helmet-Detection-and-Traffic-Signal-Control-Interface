import streamlit as st
import cv2
import numpy as np
from PIL import Image
import predict
import tempfile
import threading

def load_image(image_file):
    img = Image.open(image_file)
    return img

def process_frame(frame, frame_window):
    q1, processed_frame = predict.predict_without_helmet(frame)
    frame_window.image(processed_frame, channels='BGR')

st.title("Helmet Detection and Traffic Signal Control")

menu = ["Home", "Upload Video", "Upload Image"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("Webcam Live Feed")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.warning("Failed to read from the camera.")
            break

        # Resize the frame to reduce processing time
        frame = cv2.resize(frame, (640, 480))

        # Use threading to process the frame
        threading.Thread(target=process_frame, args=(frame, FRAME_WINDOW)).start()

    camera.release()

elif choice == "Upload Video":
    st.subheader("Upload Your Video")
    video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        vf = cv2.VideoCapture(tfile.name)
        
        stframe = st.empty()
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break

            # Resize the frame to reduce processing time
            frame = cv2.resize(frame, (640, 480))

            q1, processed_frame = predict.predict_without_helmet(frame)
            stframe.image(processed_frame, channels='BGR')

        vf.release()

elif choice == "Upload Image":
    st.subheader("Upload Your Image")
    image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if image_file:
        img = load_image(image_file)
        img_array = np.array(img)
        q1, processed_image = predict.predict_without_helmet(img_array)
        st.image(processed_image, caption='Processed Image', use_column_width=True)
