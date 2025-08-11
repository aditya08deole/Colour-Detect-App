from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import pandas as pd
import numpy as np
from io import BytesIO
import os
from functools import lru_cache

# --- Dependency ---
# Using lru_cache is a simple way to ensure the data is loaded only once.
@lru_cache()
def get_color_data():
    """
    Loads color data from CSV. This function is cached so the file is read only once.
    This is a robust way to handle "startup" data loading in a serverless environment.
    """
    index = ["color", "color_name", "hex", "R", "G", "B"]
    
    # **FIX:** The path is now relative to this file's location, looking in the SAME folder.
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, 'colors.csv')
    
    try:
        df = pd.read_csv(csv_path, names=index, header=None)
        # Pre-calculate the RGB values into a NumPy array for faster computation
        df['rgb_values'] = df[['R', 'G', 'B']].values.tolist()
        return df
    except FileNotFoundError:
        raise RuntimeError(f"Critical Error: colors.csv not found. Ensure it is in the 'api/' directory.")

# Initialize the FastAPI app
app = FastAPI()

# --- Helper Function ---
def find_closest_color(r, g, b, color_data: pd.DataFrame):
    """
    Calculates the most matching color name from the dataset using vectorized operations.
    """
    clicked_color = np.array([r, g, b])
    dataset_colors = np.array(color_data['rgb_values'].tolist())
    distances = np.sqrt(np.sum((dataset_colors - clicked_color) ** 2, axis=1))
    min_index = np.argmin(distances)
    
    closest_color = color_data.iloc[min_index]
    cname = closest_color["color_name"]
    hex_value = closest_color["hex"]
    
    return cname, hex_value

# --- API Endpoints ---

@app.post("/api/detect")
async def detect_color(
    image: UploadFile = File(...), 
    x: int = Form(...), 
    y: int = Form(...),
    color_data: pd.DataFrame = Depends(get_color_data)
):
    """
    API endpoint to detect the color of a pixel in an uploaded image.
    """
    try:
        contents = await image.read()
        img = Image.open(BytesIO(contents)).convert('RGB')

        if not (0 <= x < img.width and 0 <= y < img.height):
            raise HTTPException(status_code=400, detail="Coordinates are out of image bounds.")

        r, g, b = img.getpixel((x, y))
        color_name, hex_code = find_closest_color(r, g, b, color_data)
        
        return JSONResponse(content={
            "color_name": color_name,
            "hex_code": str(hex_code),
            "rgb": f"{r}, {g}, {b}",
            "detected_rgb": f"rgb({r},{g},{b})"
        })
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/{full_path:path}", response_class=HTMLResponse)
async def read_root(full_path: str):
    """
    Serves the main HTML page and other static files.
    This catch-all route ensures that your frontend is always served.
    """
    # Default to serving index.html
    file_path = 'index.html'
    
    script_dir = os.path.dirname(__file__)
    static_file_path = os.path.join(script_dir, '..', 'static', file_path)
    
    try:
        with open(static_file_path, "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>404 - Not Found</h1><p>The requested file could not be found.</p>", status_code=404)

