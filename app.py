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

@st.cache_data
def load_color_data():
    """
    Loads the color data from the CSV file.
    This function is cached to avoid reloading the data on every interaction.
    """
    try:
        # Define column names for the CSV file
        index = ["color", "color_name", "hex", "R", "G", "B"]
        # Read the CSV file, ensuring no header is inferred and using specified names
        csv = pd.read_csv('colors.csv', names=index, header=None)
        return csv
    except FileNotFoundError:
        st.error("Error: 'colors.csv' not found. Please make sure the file is in the same directory as the app.")
        return None

def get_color_name(R, G, B, csv_data):
    """
    Calculates the most matching color name from the dataset based on RGB values.
    It finds the color with the minimum distance in the RGB color space.
    """
    if csv_data is None:
        return "Color data not available", "#FFFFFF"

    # Vectorized calculation for performance
    # Calculate the sum of absolute differences for all colors at once
    distances = np.abs(csv_data[["R", "G", "B"]] - np.array([R, G, B])).sum(axis=1)
    
    # Find the index of the minimum distance
    min_index = distances.idxmin()
    
    # Get the color name and hex value from that index
    cname = csv_data.loc[min_index, "color_name"]
    hex_value = csv_data.loc[min_index, "hex"]
    
    return cname, hex_value

# --- Main Application UI ---

st.title("🎨 Interactive Color Detection Dashboard")
st.markdown("Upload an image, then click anywhere on it to identify the color of that pixel. The app will find the closest matching color name from our dataset.")

# Load the color dataset
color_data = load_color_data()

# --- Image Upload and Canvas ---
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Set a default image if none is uploaded
if uploaded_file is None:
    try:
        bg_image = Image.open('colorpic.jpg').convert('RGB')
        st.sidebar.info("Showing default image. Upload your own to get started!")
    except FileNotFoundError:
        st.sidebar.error("Default image 'colorpic.jpg' not found.")
        # Create a grey placeholder if the default image is missing
        bg_image = Image.new('RGB', (600, 400), (230, 230, 230)) 
else:
    bg_image = Image.open(uploaded_file).convert('RGB')

# Resize image for display if it's too large, preserving aspect ratio
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

# Create a drawable canvas to display the image and capture clicks
# We pass the image as a background to the canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0.3)",  # Fixed fill color with some opacity
    stroke_width=0,
    stroke_color="#000000",
    background_image=bg_image, # Pass the PIL image directly
    update_streamlit=True,
    height=bg_image.height,
    width=bg_image.width,
    drawing_mode="point",
    key="canvas",
)

# --- Color Detection Logic ---
if canvas_result.json_data is not None and canvas_result.json_data["objects"]:
    # Get the coordinates of the last point drawn on the canvas
    point = canvas_result.json_data["objects"][-1]
    x, y = int(point["left"]), int(point["top"])

    # Ensure coordinates are within image bounds
    if 0 <= x < bg_image.width and 0 <= y < bg_image.height:
        # Get the RGB value of the clicked pixel
        r, g, b = bg_image.getpixel((x, y))

        # Get the color name and hex code
        color_name, hex_code = get_color_name(r, g, b, color_data)

        # --- Display Results ---
        st.subheader("Detected Color Information")
        
        col1, col2 = st.columns([1, 4])

        with col1:
            # Display a box with the detected color
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
