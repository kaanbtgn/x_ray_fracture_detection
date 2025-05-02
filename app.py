import os
import logging
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(title="X-Ray Fracture Analyzer")

# Constants
MODEL_PATH = "models/best_model.keras"
THRESHOLD_PATH = "models/best_threshold.txt"
METRICS_PATH = "models/training_metrics.json"
DEFAULT_THRESHOLD = 0.5  # Fallback threshold if optimal threshold not found

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure required directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def load_model():
    """Load the trained model."""
    try:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=500,
                detail=f"Model file '{MODEL_PATH}' not found. Please train the model first."
            )
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )

def load_threshold():
    """Load the optimal threshold calculated during training."""
    try:
        if not os.path.exists(THRESHOLD_PATH):
            logger.warning(f"Optimal threshold file not found at {THRESHOLD_PATH}, using default threshold")
            return DEFAULT_THRESHOLD
        
        # Read the threshold value from the file
        with open(THRESHOLD_PATH, 'r') as f:
            threshold = float(f.read().strip())
        
        logger.info(f"Loaded optimal threshold from training: {threshold}")
        return threshold
    except Exception as e:
        logger.error(f"Error loading threshold: {e}")
        return DEFAULT_THRESHOLD

def load_metrics():
    """Load model training metrics."""
    try:
        if not os.path.exists(METRICS_PATH):
            logger.warning(f"Metrics file not found at {METRICS_PATH}")
            return {
                "accuracy": "N/A",
                "precision": "N/A",
                "recall": "N/A",
                "auc": "N/A",
                "f1_score": "N/A"
            }
        
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        
        logger.info(f"Loaded model metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error loading metrics: {e}")
        return {
            "accuracy": "N/A",
            "precision": "N/A",
            "recall": "N/A",
            "auc": "N/A",
            "f1_score": "N/A"
        }

def preprocess_image(image: Image.Image):
    """Preprocess image for model input."""
    try:
        # Resize and convert to RGB
        img = image.convert("RGB").resize((224, 224))
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        # Add batch dimension
        return img_array[None, ...]
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to preprocess image: {str(e)}"
        )

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main page."""
    threshold = load_threshold()
    metrics = load_metrics()
    logger.info(f"Current threshold value: {threshold}")
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "threshold": threshold,
            "metrics": metrics
        }
    )

@app.post("/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    """Analyze X-ray image for fractures."""
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)
        
        # Get prediction
        model = load_model()
        threshold = load_threshold()
        prediction = model.predict(processed_image, verbose=0)
        probability = float(prediction[0][0])
        
        # Create result
        result = {
            "probability": probability,
            "threshold": threshold,
            "prediction": "Fracture" if probability >= threshold else "No Fracture",
            "confidence": f"{(probability if probability >= threshold else 1 - probability) * 100:.2f}%"
        }
        
        logger.info(f"Analysis result: {result}")
        return JSONResponse(content=result)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/model-info")
async def model_info():
    """Get model information."""
    try:
        model = load_model()
        threshold = load_threshold()
        metrics = load_metrics()
        return {
            "model_type": "DenseNet121",
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "trainable_params": model.count_params(),
            "threshold": threshold,
            "threshold_source": "optimal" if os.path.exists(THRESHOLD_PATH) else "default",
            "metrics": metrics
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
