import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
import pandas as pd
import time
import streamlitpatch
import shutil


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
            with st.spinner("Processing input..."):
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
        temp_dir = tempfile.mkdtemp()
        input_video_path = os.path.join(temp_dir, "input_video"+os.path.splitext(filename)[1])
        with open (input_video_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.subheader("Original Video")
        st.video(input_video_path)

        
        if st.button("Detect Objects on Video"):
            with st.spinner("Processing video, please be aware this may take awhile"):
                results = model.track(
                    source=input_video_path,
                    conf=confidence_threshold,
                    save=True,
                    project=temp_dir,
                    name="output"
                )
                output_dir = os.path.join(temp_dir, "output")
                processed_video = None

                for file in os.listdir(output_dir):
                    if file.endswith(".mp4"):
                        processed_video = os.path.join(output_dir, file)
                        break
                if processed_video and os.path.exists(processed_video):
                    st.subheader("Processed Video")
                    st.session_state['processed_video'] = processed_video

                    st.video(processed_video)
            # ----------- Extract Detection Data and Display Table -----------
            dfs = []
            for frame_idx, res in enumerate(results):
                df_frame = res.to_df()
                df_frame["frame"] = frame_idx
                dfs.append(df_frame)

            if dfs:
                df_all = pd.concat(dfs, ignore_index=True)
                filtered = df_all[
                    df_all.track_id.notna() &
                    (df_all.confidence > confidence_threshold)
                ].copy()

                cap = cv2.VideoCapture(input_video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                if not filtered.empty:
                    df_sum = (
                        filtered
                        .groupby(['track_id', 'name'], as_index=False)
                        .agg(
                            first_frame=('frame', 'min'),
                            last_frame=('frame', 'max'),
                            confidence_score=('confidence', 'mean'),
                        )
                    )

                    df_sum['appear_from'] = pd.to_timedelta(df_sum.first_frame / fps, unit='s')
                    df_sum['appear_to'] = pd.to_timedelta(df_sum.last_frame / fps, unit='s')

                    df_sum = (
                        df_sum
                        .rename(columns={'name': 'type'})
                        [['track_id', 'type', 'appear_from', 'appear_to', 'confidence_score']]
                    )

                    df_sum['appear_from'] = (
                        df_sum['appear_from']
                        .dt.total_seconds()
                        .apply(lambda s: time.strftime("%H:%M:%S", time.gmtime(s)))
                    )
                    df_sum['appear_to'] = (
                        df_sum['appear_to']
                        .dt.total_seconds()
                        .apply(lambda s: time.strftime("%H:%M:%S", time.gmtime(s)))
                    )
                    df_sum['confidence_score'] = df_sum['confidence_score'].round(2)

                    st.write("Detected object types:", df_all['name'].unique())
                    st.dataframe(df_sum)
