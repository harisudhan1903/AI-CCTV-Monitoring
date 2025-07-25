import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import Counter
import plotly.express as px

st.set_page_config(page_title="AI CCTV Monitoring", layout="wide")

# Custom CSS styling for futuristic dark theme with neon accents
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Montserrat', sans-serif;
            background: linear-gradient(135deg, #0f0f0f, #1a1a1a);
            color: #fff;
        }

        .main-title {
            text-align: center;
            font-size: 50px;
            color: #0ff;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .sub-title {
            text-align: center;
            font-size: 18px;
            color: #888;
            margin-bottom: 30px;
        }

        .stButton>button {
            font-size: 18px;
            padding: 12px 30px;
            border-radius: 10px;
            background: linear-gradient(90deg, #f0f, #0ff);
            color: #fff;
            font-weight: bold;
            border: none;
            box-shadow: 0 0 20px rgba(255, 0, 255, 0.4);
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            box-shadow: 0 0 40px rgba(0, 255, 255, 0.6);
        }

        .status-indicator {
            font-size: 16px;
            color: lime;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔮 AI CCTV Monitoring Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Smart City Surveillance | YOLOv8 Live Object Detection</div>', unsafe_allow_html=True)

# Load YOLO model
model = YOLO("yolov8s.pt")

# Sidebar controls
st.sidebar.title("🛠️ Control Panel")
input_type = st.sidebar.radio("Select Mode", ["Upload Image", "Live Webcam"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.25, 1.0, 0.5, 0.05)

if input_type == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_np = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Detecting..."):
            results = model.predict(image, conf=confidence_threshold)
            result_img = results[0].plot()
            st.image(result_img, channels="BGR", caption="Detection Result", use_column_width=True)
            labels = [model.model.names[int(cls)] for cls in results[0].boxes.cls]
            count = Counter(labels)
            st.success("Detection Complete")
            if count:
                st.subheader("📊 Object Statistics")
                chart = px.bar(x=list(count.keys()), y=list(count.values()), color=list(count.keys()),
                               labels={'x': 'Class', 'y': 'Count'}, color_discrete_sequence=px.colors.sequential.Rainbow)
                st.plotly_chart(chart, use_container_width=True)

elif input_type == "Live Webcam":
    start = st.button("▶️ Start Webcam")
    stop = st.button("⛔ Stop Webcam")
    stframe = st.empty()
    cap = None

    if start:
        cap = cv2.VideoCapture(0)
        stats_placeholder = st.empty()
        object_counter = Counter()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=confidence_threshold)
            annotated = results[0].plot()
            classes = [model.model.names[int(c)] for c in results[0].boxes.cls]
            object_counter.update(classes)

            stframe.image(annotated, channels="BGR", use_column_width=True)

            with stats_placeholder.container():
                if object_counter:
                    st.subheader("📊 Live Object Stats")
                    chart = px.bar(x=list(object_counter.keys()), y=list(object_counter.values()), color=list(object_counter.keys()),
                                   labels={'x': 'Class', 'y': 'Count'}, color_discrete_sequence=px.colors.qualitative.Vivid)
                    st.plotly_chart(chart, use_container_width=True)

            if stop:
                break

        cap.release()
        st.success("✅ Webcam stopped")