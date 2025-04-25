# X-Ray Fracture Detection System

An advanced medical imaging analysis system that uses deep learning to detect fractures in X-ray images. The system provides comprehensive analysis and reporting features through an intuitive graphical interface.

## Features

### 1. X-ray Analysis
- Real-time fracture detection using DenseNet-121 model
- Confidence scoring and probability assessment
- Image quality analysis metrics
- Pixel intensity distribution analysis

### 2. Comprehensive Reporting
- Detailed analysis reports with multiple visualization options
- Image quality metrics (entropy, intensity statistics)
- Technical metadata logging
- Historical analysis logging and statistics

### 3. Model Training and Evaluation
- DenseNet-121 based architecture
- Automatic loss selection (BinaryFocalCrossentropy/BinaryCrossentropy)
- Comprehensive training metrics (accuracy, AUC, recall, precision)
- Early stopping and learning rate reduction
- Mixed precision training support

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd xray_detection
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.10 or higher
- TensorFlow 2.19 or higher
- Required packages are listed in `requirements.txt`

## Usage

### Training the Model
```bash
./train.sh
```
This will:
- Train the DenseNet-121 model on your dataset
- Save the best model as `xray_dense121_best.keras`
- Generate training logs in `training.log`

### Running the Application
```bash
./run.sh
```
This will:
- Launch the application
- Access the GUI through your web browser at `http://localhost:8501`

## Project Structure

```
xray_detection/
├── app.py                 # Main application
├── train_eval_tf.py       # Training and evaluation script
├── create_labels.py       # Dataset label creation utility
├── requirements.txt       # Package dependencies
├── train.sh              # Training script
├── run.sh                # Application startup script
├── README.md             # Project documentation
├── README.dataset.txt    # Dataset information
├── usage_instructions.txt # Usage guide
├── static/               # Static web assets
├── templates/            # Web templates
├── runs/                 # Model checkpoints
└── logs/                 # Analysis logs
```

## Model Information

The system uses a DenseNet-121 based deep learning model with the following features:
- Pre-trained on ImageNet
- Fine-tuned for fracture detection
- Automatic loss selection based on class imbalance
- Regularization: augmentation + L2 + dropout + early-stopping
- Mixed precision training support

## Development

To contribute to the project:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

[Your chosen license]