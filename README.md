# X-Ray Fracture Detection System

An advanced medical imaging analysis system that uses deep learning to detect fractures in X-ray images. The system provides comprehensive analysis and reporting features through an intuitive graphical interface.

## Features

### 1. X-ray Analysis
- Real-time fracture detection using deep learning
- Confidence scoring and probability assessment
- Activation heatmap visualization
- Image quality analysis metrics
- Pixel intensity distribution analysis

### 2. Comprehensive Reporting
- Detailed analysis reports with multiple visualization options
- Image quality metrics (entropy, intensity statistics)
- Technical metadata logging
- Interactive visualizations using Plotly
- Historical analysis logging and statistics

### 3. Model Performance Analysis
- Model architecture visualization
- Training history and metrics plotting
- Performance statistics and evaluation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd xray_detection
```

2. Run the start script:
```bash
chmod +x start.sh
./start.sh
```

The script will:
- Create a Python virtual environment
- Install all required dependencies
- Launch the Streamlit GUI

## Requirements

- Python 3.10 or higher
- macOS with Apple Silicon (M1/M2) support
- Required packages are listed in `requirements.txt`

## Usage

1. Launch the application:
```bash
./start.sh
```

2. Access the GUI through your web browser at `http://localhost:8501`

3. Select operation mode:
   - **Analyse X-ray**: Upload and analyze X-ray images
   - **View logs**: Review past analyses and statistics
   - **Model Performance**: Examine model architecture and metrics

4. For analysis:
   - Upload an X-ray image (supported formats: PNG, JPG, JPEG)
   - Click "Analyse" to process the image
   - View comprehensive results in the tabbed interface

## Project Structure

```
xray_detection/
├── gui.py              # Main GUI application
├── requirements.txt    # Package dependencies
├── start.sh           # Startup script
├── runs/              # Model checkpoints
└── logs/              # Analysis logs
```

## Model Information

The system uses a DenseNet-based deep learning model trained on X-ray images. The model outputs:
- Binary classification (fracture/no fracture)
- Confidence scores
- Activation heatmaps for interpretability

## Logging

Analysis results are automatically logged to `logs/inference.csv` with the following information:
- Timestamp
- Filename
- Detection result
- Confidence score
- Image quality metrics

## Development

To contribute to the project:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

[Your chosen license]