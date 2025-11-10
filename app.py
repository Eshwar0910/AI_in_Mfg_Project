import streamlit as st
import numpy as np
import os
import requests
try:
    import cv2
    CV2_AVAILABLE = True
except ModuleNotFoundError:
    cv2 = None
    CV2_AVAILABLE = False
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Page Config
st.set_page_config(page_title="Defect Detector", layout="centered")

# Custom CSS for elegant black-and-white UI
st.markdown("""
    <style>
        body {
            background-color: #111;
            color: #fff;
        }
        .stApp {
            background-color: #000;
            color: white;
            font-family: 'Roboto', sans-serif;
        }
        .main-title {
            text-align: center;
            font-size: 2.3rem;
            font-weight: 700;
            margin-bottom: 15px;
        }
        .sub-title {
            text-align: center;
            font-size: 1rem;
            color: #ccc;
            margin-bottom: 25px;
        }
        .detect-button > button {
            background-color: white !important;
            color: black !important;
            border-radius: 8px !important;
            font-weight: bold !important;
            transition: 0.3s !important;
        }
        .detect-button > button:hover {
            background-color: #888 !important;
            color: #000 !important;
        }
        .uploaded-img {
            border: 2px solid white;
            border-radius: 10px;
            padding: 5px;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

def ensure_model_present():
    """Ensure the model file exists locally. If not, try to download from MODEL_URL env var.

    If neither the file nor MODEL_URL is available, stop the app with an explanatory message.
    """
    model_path = os.environ.get("MODEL_PATH", "mobilenetv2_finetuned_best_model.keras")
    model_url = os.environ.get("MODEL_URL")

    if os.path.exists(model_path):
        return model_path

    if not model_url:
        st.error(f"Model file '{model_path}' not found and MODEL_URL is not set.\n\n" \
                 "Upload the model to the repo, set MODEL_URL in Streamlit Cloud, or mount the model into the container.")
        st.stop()

    # Attempt to download the model
    try:
        with st.spinner("Downloading model from MODEL_URL..."):
            resp = requests.get(model_url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        st.success("Model downloaded successfully.")
        return model_path
    except Exception as e:
        st.error(f"Failed to download model from MODEL_URL: {e}")
        st.stop()


# Load model (cached for efficiency)
@st.cache_resource
def load_defect_model(model_path: str):
    return load_model(model_path)


# Ensure model is present (either in repo or downloaded)
MODEL_PATH = ensure_model_present()
model = load_defect_model(MODEL_PATH)

# Title
st.markdown("<div class='main-title'>⚙️ Defect Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Upload an image and click 'Detect Defect' to analyze</div>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_container_width=True, output_format="PNG", channels="RGB")

    # Center detect button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        detect_clicked = st.container()
        with detect_clicked:
            st.markdown('<div class="detect-button">', unsafe_allow_html=True)
            detect = st.button("Detect Defect")
            st.markdown('</div>', unsafe_allow_html=True)

    if detect:
        with st.spinner("Analyzing image..."):
            # Preprocess
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = model.predict(img_array)
            is_defective = prediction[0][0] > 0.7

            if is_defective:
                st.success("✅ No Defects Detected!")
            else:
                st.warning("⚠️ Defective Detected!")

                if CV2_AVAILABLE:
                    st.info("Highlighting defect areas with OpenCV...")

                    # Convert to OpenCV format
                    opencv_img = np.array(img)
                    gray = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray, 100, 200)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        if w > 30 and h > 30:
                            cv2.rectangle(opencv_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

                    st.image(opencv_img, caption="Detected Defects", use_container_width=True, channels="RGB")
                else:
                    st.error("OpenCV (`cv2`) is not available in the Python environment running Streamlit. Defect highlighting is disabled.")
                    st.markdown(
                        "To enable highlighting, either activate your project's virtual environment before running Streamlit or install OpenCV into the Python that runs Streamlit. Example PowerShell commands:\\n"
                        "- Activate environment (recommended): `./myenv/Scripts/Activate.ps1; streamlit run app.py`\\n"
                        "- Or install into the current Python: `pip install opencv-python`\n"
                    )
