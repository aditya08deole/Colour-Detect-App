import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import base64
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(
    page_title="Interactive Color Detector",
    page_icon="🎨",
    layout="wide",
)

# --- Helper Functions ---

def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a Base64 encoded data URL."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@st.cache_data
def load_color_data():
    """
    Loads the color data from the CSV file.
    This function is cached to avoid reloading the data on every interaction.
    """
    try:
        index = ["color", "color_name", "hex", "R", "G", "B"]
        csv = pd.read_csv('colors.csv', names=index, header=None)
        return csv
    except FileNotFoundError:
        st.error("Error: 'colors.csv' not found. Please make sure the file is in the same directory as the app.")
        return None

def get_color_name(R, G, B, csv_data):
    """
    Calculates the most matching color name from the dataset based on RGB values.
    """
    if csv_data is None:
        return "Color data not available", "#FFFFFF"

    distances = np.abs(csv_data[["R", "G", "B"]] - np.array([R, G, B])).sum(axis=1)
    min_index = distances.idxmin()
    cname = csv_data.loc[min_index, "color_name"]
    hex_value = csv_data.loc[min_index, "hex"]
    return cname, hex_value

# --- Main Application UI ---

st.title("🎨 Interactive Color Detection Dashboard")
st.markdown("Upload an image (or drag and drop one), then click anywhere on it to identify the color of that pixel.")

# Load the color dataset
color_data = load_color_data()

# --- Image Upload and Canvas ---
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    try:
        bg_image = Image.open('colorpic.jpg').convert('RGB')
        st.sidebar.info("Showing default image. Upload your own to get started!")
    except FileNotFoundError:
        st.sidebar.error("Default image 'colorpic.jpg' not found.")
        bg_image = Image.new('RGB', (600, 400), (230, 230, 230)) 
else:
    bg_image = Image.open(uploaded_file).convert('RGB')

# Resize image for display if it's too large
max_size = 800
width, height = bg_image.size
if width > max_size or height > max_size:
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    bg_image = bg_image.resize((new_width, new_height))

st.subheader("Click on the Image to Detect a Color")

# **FIX:** Convert the image to a Base64 data URL to prevent errors
background_image_url = image_to_base64(bg_image)

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0.3)",
    stroke_width=0,
    background_image=background_image_url, # Use the reliable data URL
    update_streamlit=True,
    height=bg_image.height,
    width=bg_image.width,
    drawing_mode="point",
    key="canvas",
)

# --- Color Detection Logic ---
if canvas_result.json_data is not None and canvas_result.json_data["objects"]:
    point = canvas_result.json_data["objects"][-1]
    x, y = int(point["left"]), int(point["top"])

    if 0 <= x < bg_image.width and 0 <= y < bg_image.height:
        r, g, b = bg_image.getpixel((x, y))
        color_name, hex_code = get_color_name(r, g, b, color_data)

        st.subheader("Detected Color Information")
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(
                f"""
                <div style="
                    width: 100px;
                    height: 100px;
                    background-color: rgb({r}, {g}, {b});
                    border: 2px solid #d3d3d3;
                    border-radius: 10px;
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                "></div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.metric(label="**Closest Color Name**", value=color_name)
            st.metric(label="**Hex Code**", value=str(hex_code))
            st.metric(label="**RGB Value**", value=f"{r}, {g}, {b}")
else:
    st.info("Click on a point on the image to see its color details.")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit.")
