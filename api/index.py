from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import os

app = FastAPI()

# --- Helper Functions ---

@app.on_event("startup")
async def startup_event():
    """Load data on startup and cache it."""
    global color_data
    index = ["color", "color_name", "hex", "R", "G", "B"]
    # Construct path relative to the current file
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, '..', 'static', 'colors.csv')
    try:
        color_data = pd.read_csv(csv_path, names=index, header=None)
    except FileNotFoundError:
        color_data = None
        print(f"Error: Could not find colors.csv at {csv_path}")


def get_color_name(R, G, B):
    """Calculates the most matching color name from the dataset."""
    if color_data is None:
        return "Color data not available", "#FFFFFF"

    distances = np.abs(color_data[["R", "G", "B"]] - np.array([R, G, B])).sum(axis=1)
    min_index = distances.idxmin()
    cname = color_data.loc[min_index, "color_name"]
    hex_value = color_data.loc[min_index, "hex"]
    return cname, hex_value

# --- API Endpoints ---

@app.post("/api/detect")
async def detect_color(image: UploadFile = File(...), x: int = Form(...), y: int = Form(...)):
    """
    API endpoint to detect the color of a pixel in an uploaded image.
    """
    try:
        # Read the uploaded image
        contents = await image.read()
        img = Image.open(BytesIO(contents)).convert('RGB')

        # Ensure coordinates are within image bounds
        if 0 <= x < img.width and 0 <= y < img.height:
            r, g, b = img.getpixel((x, y))
            color_name, hex_code = get_color_name(r, g, b)
            
            return JSONResponse(content={
                "color_name": color_name,
                "hex_code": str(hex_code),
                "rgb": f"{r}, {g}, {b}",
                "detected_rgb": f"rgb({r},{g},{b})"
            })
        else:
            return JSONResponse(status_code=400, content={"error": "Coordinates are out of bounds."})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the main HTML page."""
    # Construct path to the HTML file
    script_dir = os.path.dirname(__file__)
    html_path = os.path.join(script_dir, '..', 'static', 'index.html')
    try:
        with open(html_path, "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found.</h1>", status_code=500)

