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
import glob

@st.cache_resource
def load_model():
    model_path = 'archive/run14/train14/weights/best.pt'
    return YOLO(model_path)


model = load_model()

st.title("Automated Pothole Detection")
if not st.session_state.get("authenticated", False):
    st.warning("You must login first. Please use the Login page.")
    st.stop()
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"])

confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

if uploaded_file is not None:
    filename = uploaded_file.name.lower()

    if filename.endswith(("jpg", "jpeg", "png")):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        original_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        st.image(original_rgb, caption="Original Image", use_container_width=True)
        
        if st.button("Detect Objects on Image"):
            with st.spinner("Processing input..."):
                results = model.predict(source=img_cv, conf=confidence_threshold)
                res = results[0]
                
                annotated_img = res.plot()
                annotated_rgb = annotated_img[..., ::-1] 
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_rgb, caption="Original Image", use_container_width=True)
                with col2:
                    st.image(annotated_rgb, caption="Segmented", use_container_width=True)


    elif filename.endswith(("mp4", "mov", "avi", "mkv")):
            if 'temp_dir' not in st.session_state:
                st.session_state.temp_dir = tempfile.mkdtemp()
                st.session_state.results_dir = os.path.join(st.session_state.temp_dir, "results")
                os.makedirs(st.session_state.results_dir, exist_ok=True)
            input_name = f"input_{int(time.time())}"
            input_ext = os.path.splitext(filename)[1]
            input_video_path = os.path.join(st.session_state.temp_dir, f"{input_name}{input_ext}")

            with open(input_video_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            st.subheader("Original Video")
            st.video(input_video_path)
            st.session_state.input_video_path = input_video_path

            if (st.button("Detect Objects on Video")):
                with st.spinner("Processing video... please be aware that this may take awhile."):
                    output_folder = os.path.join(st.session_state.results_dir, f"output_{int(time.time())}")
                    os.makedirs(output_folder, exist_ok=True)

                    try:
                        results = model.track(
                            source = input_video_path,
                            conf=confidence_threshold,
                            save=True,
                            project=st.session_state.results_dir,
                            name=f"output_{int(time.time())}"
                        )

                        st.write(f"YOLO Output Directory: {results[0].save_dir}")

                        output_dir = results[0].save_dir
                        processed_video = glob.glob(os.path.join(output_dir, "*.mp4"))

                        if processed_video:
                            processed_video = processed_video[0]
                            st.session_state.processed_video = processed_video
                            
                            output_copy = os.path.join(st.session_state.results_dir, "latest_output,mp4")
                            shutil.copy2(processed_video, output_copy)
                            st.session_state.output_copy = output_copy

                            st.header("Processed Video")
                            st.video(output_copy)

                            with open(output_copy, 'rb') as video_file:
                                video_bytes = video_file.read()
                                st.subheader("Secondary Video Display")
                                st.video(video_bytes)
                        else:
                            st.error("No processed video found in YOLO output directory")
                            
                    except Exception as e:
                        st.error(f"Error during video processing: {str(e)}")



            # ----------- Extract Detection Data and Display Table -----------
                    try:
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
                    except Exception as e:
                        st.error(f"Error processing detection data {str(e)}")
