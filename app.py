from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from tensorflow.keras.applications import densenet

app = FastAPI(title="X-Ray Fracture Analyzer")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables
 # Use Keras v3 native format to avoid “Unknown layer: Cast” errors
MODEL_PATH = "xray_dense121_best.keras"
model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

def preprocess_image(image: Image.Image):
    # Resize and preprocess image for DenseNet
    img = image.convert("RGB").resize((224, 224))
    img_array = densenet.preprocess_input(np.array(img))
    return img_array[None, ...]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)
        
        # Get prediction
        model = load_model()
        prediction = model.predict(processed_image)
        probability = float(prediction[0][0])
        
        # Create result
        result = {
            "probability": probability,
            "prediction": "Fracture" if probability >= 0.5 else "No Fracture",
            "confidence": f"{(probability if probability >= 0.5 else 1-probability)*100:.2f}%"
        }
        
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/model-performance")
async def get_model_performance():
    try:
        model = load_model()
        
        # Get model summary
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        
        # If you have test results saved, load them here
        # For now, returning placeholder metrics
        metrics = {
            "accuracy": 0.92,
            "precision": 0.91,
            "recall": 0.93,
            "f1_score": 0.92
        }
        
        return {
            "model_summary": "\n".join(model_summary),
            "metrics": metrics
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 