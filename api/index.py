from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import pandas as pd
import numpy as np
from io import BytesIO
import os
from functools import lru_cache
import time
from collections import defaultdict

# --- Security & Performance Globals ---
request_counts = defaultdict(lambda: {'count': 0, 'timestamp': 0})
RATE_LIMIT_SECONDS = 5
MAX_REQUESTS = 10
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"]

# --- Dependency with Speed Optimizations ---
@lru_cache()
def get_color_data():
    """
    Loads color data and pre-computes a NumPy array for ultra-fast lookups.
    This function is cached, so it runs only once.
    """
    index = ["color", "color_name", "hex", "R", "G", "B"]
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, 'colors.csv')
    try:
        df = pd.read_csv(csv_path, names=index, header=None)
        # **SPEED OPTIMIZATION 1:** Pre-compute the color dataset into a NumPy array.
        # This avoids converting from pandas on every API call.
        color_dataset_np = df[['R', 'G', 'B']].to_numpy(dtype=np.int16)
        return df, color_dataset_np
    except FileNotFoundError:
        raise RuntimeError(f"Critical Error: colors.csv not found at {csv_path}")

app = FastAPI()

# --- Helper Function with Speed Optimizations ---
def find_closest_color(r, g, b, color_df: pd.DataFrame, color_dataset_np: np.ndarray):
    """
    Calculates the closest color using squared Euclidean distance for maximum speed.
    """
    clicked_color = np.array([r, g, b], dtype=np.int16)
    
    # **SPEED OPTIMIZATION 2:** Calculate squared Euclidean distance.
    # This avoids the computationally expensive square root operation.
    distances_sq = np.sum((color_dataset_np - clicked_color) ** 2, axis=1)
    
    min_index = np.argmin(distances_sq)
    closest_color = color_df.iloc[min_index]
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
    # Unpack the cached data
    color_df, color_dataset_np = cached_data

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
        color_name, hex_code = find_closest_color(r, g, b, color_df, color_dataset_np)
        
        return JSONResponse(content={
            "color_name": color_name,
            "hex_code": hex_code,
            "rgb": f"{r}, {g}, {b}",
            "detected_rgb": f"rgb({r},{g},{b})"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_frontend(full_path: str):
    """Serves the main HTML page."""
    script_dir = os.path.dirname(__file__)
    static_file_path = os.path.join(script_dir, '..', 'static', 'index.html')
    try:
        with open(static_file_path, "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>404 - Not Found</h1>", status_code=404)
