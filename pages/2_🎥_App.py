import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
import pandas as pd
import time
import streamlitpatch


# Cache the model to avoid reloading on every run
@st.cache_resource
def load_model():
    model_path = 'archive/run14/train14/weights/best.pt'
    return YOLO(model_path)


model = load_model()

st.title("Automated Pothole Detection")
if not st.session_state.get("authenticated", False):
    st.warning("You must login first. Please use the Login page.")
    st.stop()
# Allow user to upload image or video
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"])

confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

if uploaded_file is not None:
    filename = uploaded_file.name.lower()

    # If the file is an image
    if filename.endswith(("jpg", "jpeg", "png")):
        # Convert uploaded file to a NumPy array for OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR image
        
        # Convert to RGB for display
        original_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        st.image(original_rgb, caption="Original Image", use_container_width=True)
        
        if st.button("Detect Objects on Image"):
            results = model.predict(source=img_cv, conf=confidence_threshold)
            res = results[0]
            
            # Get annotated image (res.plot() returns a BGR image)
            annotated_img = res.plot()
            annotated_rgb = annotated_img[..., ::-1]  # convert BGR to RGB
            
            # Display images side by side using Streamlit columns
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_rgb, caption="Original Image", use_container_width=True)
            with col2:
                st.image(annotated_rgb, caption="Segmented", use_container_width=True)

    elif filename.endswith(("mp4", "mov", "avi", "mkv")):
        tfile_bytes = uploaded_file.read()
        temp_vid = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_vid.write(tfile_bytes)
        temp_vid.flush()
        temp_vid_path = temp_vid.name

        cap = cv2.VideoCapture(temp_vid_path)
        stframe = st.empty()
        detections = []
        frame_number = 0

        if st.button("Detect Objects on Video"):
            fps = cap.get(cv2.CAP_PROP_FPS)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(source=frame, conf=confidence_threshold)
                result = results[0]  # YOLOv11 returns list-like results

                # Draw and display
                annotated_frame = result.plot()
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
                #time.sleep(speed / 1000.0)

                # Collect detections
                boxes = result.boxes
                if boxes is not None and hasattr(boxes, 'cls'):
                    for cls_id, conf, box_coords in zip(boxes.cls, boxes.conf, boxes.xyxy):
                        detections.append({
                            "frame": frame_number,
                            "class": model.names[int(cls_id)],
                            "confidence": float(conf),
                            "xmin": float(box_coords[0]),
                            "ymin": float(box_coords[1]),
                            "xmax": float(box_coords[2]),
                            "ymax": float(box_coords[3]),
                        })

                frame_number += 1

            cap.release()
            os.remove(temp_vid_path)

            if detections:
                df = pd.DataFrame(detections)
                st.markdown("### 📊 Detection Summary")
                st.dataframe(df)
            else:
                st.warning("No objects were detected.")
