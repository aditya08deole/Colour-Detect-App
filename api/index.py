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
from skimage.color import rgb2lab

# --- Configuration via Environment Variables ---
RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", 5))
MAX_REQUESTS = int(os.getenv("MAX_REQUESTS", 10))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 5 * 1024 * 1024)) # 5 MB
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"]

request_counts = defaultdict(lambda: {'count': 0, 'timestamp': 0})

# --- Dependency with CIELAB Color Data ---
@lru_cache()
def get_color_data():
    """
    Loads color data, converts it to the perceptually uniform CIELAB space,
    and caches it for high-performance lookups.
    """
    index = ["color", "color_name", "hex", "R", "G", "B"]
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, 'colors.csv')
    try:
        df = pd.read_csv(csv_path, names=index, header=None)
        lookup_data = df[['color_name', 'hex']].to_dict('records')
        
        # Convert the entire dataset to LAB colors for accurate comparison
        rgb_colors = df[['R', 'G', 'B']].to_numpy(dtype=np.uint8)
        # Reshape for scikit-image, convert, and reshape back
        lab_color_dataset = rgb2lab(rgb_colors.reshape(1, -1, 3)).reshape(-1, 3)
        
        return lookup_data, lab_color_dataset
    except FileNotFoundError:
        raise RuntimeError("Critical Error: colors.csv not found.")

app = FastAPI()

# --- Helper Functions ---
def find_closest_color(r, g, b, lookup_data: list, lab_color_dataset: np.ndarray):
    """
    Finds the closest color name using Euclidean distance in the CIELAB space.
    """
    clicked_rgb = np.uint8([[[r, g, b]]])
    clicked_lab = rgb2lab(clicked_rgb).flatten()
    
    # Calculate Euclidean distance in LAB space
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

    # Rate Limiting
    client_ip = request.client.host
    current_time = time.time()
    if current_time - request_counts[client_ip]['timestamp'] > RATE_LIMIT_SECONDS:
        request_counts[client_ip] = {'count': 1, 'timestamp': current_time}
    else:
        request_counts[client_ip]['count'] += 1
    if request_counts[client_ip]['count'] > MAX_REQUESTS:
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
    """Generates a simple palette of shades for a given color."""
    shades = []
    tints = []
    base_color = np.array([r, g, b])
    white = np.array([255, 255, 255])
    
    # Generate 5 shades (mixing with black)
    for i in np.linspace(1, 0.2, 5):
        shade = (base_color * i).astype(int)
        shades.append(f"rgb({shade[0]},{shade[1]},{shade[2]})")
        
    # Generate 5 tints (mixing with white)
    for i in np.linspace(0.2, 1, 5)[::-1]: # Reversed for light to dark
        tint = (base_color + (white - base_color) * i).astype(int)
        tints.append(f"rgb({tint[0]},{tint[1]},{tint[2]})")

    return {"shades": shades, "tints": tints}


# --- Static File Serving ---
# Mount the 'static' directory to serve HTML, CSS, JS, etc.
static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serves the main index.html file."""
    index_path = os.path.join(static_dir, 'index.html')
    try:
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>404 - Not Found</h1>", status_code=404)