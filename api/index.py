from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import pandas as pd
import numpy as np
from io import BytesIO
import os
from functools import lru_cache
import time
from collections import defaultdict

# --- Configuration via Environment Variables ---
RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", 5))
MAX_REQUESTS = int(os.getenv("MAX_REQUESTS", 10))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 5 * 1024 * 1024)) # 5 MB
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"]

request_counts = defaultdict(lambda: {'count': 0, 'timestamp': 0})

# --- Self-Contained RGB to CIELAB Conversion Function ---
def rgb_to_lab(r, g, b):
    """Converts an RGB color to CIELAB space without scikit-image."""
    # Normalize RGB values
    rgb = np.array([r, g, b]) / 255.0
    
    # Gamma correction
    rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    
    # RGB to XYZ conversion matrix
    xyz_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = np.dot(xyz_matrix, rgb)
    
    # XYZ to CIELAB conversion (D65 reference white)
    xyz /= np.array([0.95047, 1.00000, 1.08883])
    
    # LAB transformation
    xyz = np.where(xyz > 0.008856, xyz ** (1/3), (7.787 * xyz) + (16/116))
    
    L = (116 * xyz[1]) - 16
    a = 500 * (xyz[0] - xyz[1])
    b_lab = 200 * (xyz[1] - xyz[2])
    
    return np.array([L, a, b_lab])


# --- Dependency with CIELAB Color Data ---
@lru_cache()
def get_color_data():
    """
    Loads color data, converts it to CIELAB space, and caches it.
    """
    index = ["color", "color_name", "hex", "R", "G", "B"]
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, 'colors.csv')
    try:
        df = pd.read_csv(csv_path, names=index, header=None)
        if df.empty:
            raise RuntimeError("Critical Error: colors.csv is empty or unreadable.")

        lookup_data = df[['color_name', 'hex']].to_dict('records')
        
        rgb_colors = df[['R', 'G', 'B']].to_numpy(dtype=np.uint8)
        lab_color_dataset = np.array([rgb_to_lab(r, g, b) for r, g, b in rgb_colors])
        
        return lookup_data, lab_color_dataset
    except FileNotFoundError:
        raise RuntimeError("Critical Error: colors.csv not found.")

app = FastAPI()

# --- Helper Functions ---
def find_closest_color(r, g, b, lookup_data: list, lab_color_dataset: np.ndarray):
    """Finds the closest color name using Euclidean distance in the CIELAB space."""
    clicked_lab = rgb_to_lab(r, g, b)
    
    distances = np.linalg.norm(lab_color_dataset - clicked_lab, axis=1)
    min_index = np.argmin(distances)
    
    closest_color = lookup_data[min_index]
    return closest_color["color_name"], str(closest_color["hex"])


# --- API Endpoints ---
@app.post("/api/detect")
async def detect_color(
    request: Request,
    image: UploadFile = File(...), 
    x: int = Form(...), 
    y: int = Form(...),
    cached_data: tuple = Depends(get_color_data)
):
    """API endpoint to detect the color with security and performance in mind."""
    lookup_data, lab_color_dataset = cached_data

    # UPGRADE: More robust rate limiting using IP + User-Agent
    client_identifier = f"{request.client.host}:{request.headers.get('user-agent', 'unknown')}"
    current_time = time.time()
    
    if current_time - request_counts[client_identifier]['timestamp'] > RATE_LIMIT_SECONDS:
        request_counts[client_identifier] = {'count': 1, 'timestamp': current_time}
    else:
        request_counts[client_identifier]['count'] += 1
    
    if request_counts[client_identifier]['count'] > MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Too Many Requests.")

    # File Validation
    if image.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type.")
    contents = await image.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large.")

    try:
        img = Image.open(BytesIO(contents)).convert('RGB')
        if not (0 <= x < img.width and 0 <= y < img.height):
            raise HTTPException(status_code=400, detail="Coordinates out of bounds.")

        r, g, b = img.getpixel((x, y))
        color_name, hex_code = find_closest_color(r, g, b, lookup_data, lab_color_dataset)
        
        return JSONResponse(content={
            "color_name": color_name,
            "hex_code": hex_code,
            "rgb": f"{r},{g},{b}",
            "detected_rgb": f"rgb({r},{g},{b})"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/api/palette")
def get_palette(r: int, g: int, b: int):
    """Generates a simple palette of shades and tints for a given color."""
    base_color = np.array([r, g, b])
    
    # UPGRADE: Refactored to reduce code duplication
    def generate_variant(color, mix_with, factors):
        return [
            (color + (mix_with - color) * i).astype(int)
            for i in factors
        ]

    tints = generate_variant(base_color, np.array([255, 255, 255]), np.linspace(0.2, 1, 5)[::-1])
    shades = generate_variant(base_color, np.array([0, 0, 0]), np.linspace(0, 0.8, 5))

    return {
        "shades": [f"rgb({s[0]},{s[1]},{s[2]})" for s in shades],
        "tints": [f"rgb({t[0]},{t[1]},{t[2]})" for t in tints]
    }

# --- Static Files & Frontend Serving ---
static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serves the main index.html file."""
    index_path = os.path.join(static_dir, 'index.html')
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>404 - Not Found</h1>", status_code=404)