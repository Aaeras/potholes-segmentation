import streamlitpatch
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
import pandas as pd
import time
import torch

loginPage = st.Page("pages/login.py", title="Login ", icon="🔒")
modelPage = st.Page("pages/app.py", title="App ", icon="🎥")

if "authenticated" not in st.session_state:
    pg = st.navigation(loginPage)
else:
    pg = st.navigation([loginPage, modelPage])
pg.run()