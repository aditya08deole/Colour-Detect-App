from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import pandas as pd
import numpy as np
from io import BytesIO
import os
from functools lru_cache

# --- Dependency ---
# Using lru_cache is a simple way to ensure the data is loaded only once.
@lru_cache()
def get_color_data():
    """
    Loads color data from CSV. This function is cached so the file is read only once.
    This is a robust way to handle "startup" data loading in a serverless environment.
    """
    index = ["color", "color_name", "hex", "R", "G", "B"]
    # The path is relative to this file's location, making it robust.
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, '..', 'static', 'colors.csv')
    try:
        df = pd.read_csv(csv_path, names=index, header=None)
        # Pre-calculate the RGB values into a NumPy array for faster computation
        df['rgb_values'] = df[['R', 'G', 'B']].values.tolist()
        return df
    except FileNotFoundError:
        # If the file isn't found, the app can't work. Raise a server error.
        raise RuntimeError(f"Critical Error: colors.csv not found at {csv_path}")

# Initialize the FastAPI app
app = FastAPI()

# --- Helper Function ---
def find_closest_color(r, g, b, color_data: pd.DataFrame):
    """
    Calculates the most matching color name from the dataset using vectorized operations.
    
    Args:
        r, g, b (int): The RGB values of the pixel.
        color_data (pd.DataFrame): The DataFrame containing the color dataset.
    
    Returns:
        A tuple containing the color name (str) and hex code (str).
    """
    # Create a NumPy array for the clicked color
    clicked_color = np.array([r, g, b])
    
    # Extract the pre-calculated RGB values from the DataFrame
    dataset_colors = np.array(color_data['rgb_values'].tolist())
    
    # Calculate the Euclidean distance in the RGB space.
    # This is a more accurate color distance metric than sum of absolute differences.
    distances = np.sqrt(np.sum((dataset_colors - clicked_color) ** 2, axis=1))
    
    # Find the index of the color with the minimum distance
    min_index = np.argmin(distances)
    
    # Retrieve the color name and hex code from that index
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
    color_data: pd.DataFrame = Depends(get_color_data) # Dependency Injection
):
    """
    API endpoint to detect the color of a pixel in an uploaded image.
    It uses the `get_color_data` dependency to ensure the color data is available.
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
        # Re-raise HTTP exceptions to let FastAPI handle them
        raise e
    except Exception as e:
        # For any other unexpected errors, return a generic server error
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the main HTML page."""
    script_dir = os.path.dirname(__file__)
    html_path = os.path.join(script_dir, '..', 'static', 'index.html')
    try:
        with open(html_path, "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>500 - Internal Error</h1><p>Frontend file not found.</p>", status_code=500)
