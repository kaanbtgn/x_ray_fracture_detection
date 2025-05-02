# X-Ray Fracture Detection System

An advanced deep learning system for detecting fractures in X-ray images using DenseNet-121. The system provides real-time analysis through a modern web interface, optimized for both Apple Silicon and NVIDIA GPUs.

![X-Ray Analysis Interface](docs/interface.png)

## Features

### 1. X-ray Analysis
- 🔍 Real-time fracture detection using DenseNet-121
- 📊 Confidence scoring and probability assessment
- 🎯 Optimal threshold calculation for accurate predictions
- 💻 Cross-platform support (Apple Silicon and CUDA GPUs)

### 2. Model Training
- 🧠 DenseNet-121 based architecture with transfer learning
- ⚡️ Automatic hardware detection and optimization
- 📈 Comprehensive metrics tracking:
  - Accuracy
  - AUC (Area Under Curve)
  - Precision
  - Recall
  - F1 Score
- 🔄 Early stopping and learning rate reduction
- 🔀 Data augmentation for better generalization

## Prerequisites

- Python 3.10 or higher
- For Apple Silicon Macs:
  - tensorflow-macos==2.16.1
  - tensorflow-metal==1.1.0
- For NVIDIA GPUs:
  - tensorflow==2.16.1 with CUDA support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/xray-fracture-detection.git
cd xray-fracture-detection
```

2. Create and activate virtual environment:
```bash
# On macOS/Linux
python3 -m venv venv_tf
source venv_tf/bin/activate

# On Windows
python -m venv venv_tf
.\venv_tf\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Structure

Organize your dataset in the following structure:
```
dataset/
├── train/
│   ├── images/         # Fractured X-rays
│   └── not_fractured/  # Normal X-rays
├── val/
│   ├── images/
│   └── not_fractured/
└── test/
    ├── images/
    └── not_fractured/
```

## Usage

### Training the Model

```bash
./train.sh
```

This will:
- 🔍 Detect your hardware (Apple Silicon/CUDA/CPU)
- 📊 Train the model with optimal parameters
- 💾 Save the best model to `models/best_model.keras`
- 📈 Generate training metrics in `models/training_metrics.json`
- 🎯 Calculate and save optimal threshold

### Running the Web Interface

```bash
./run.sh
```

This will:
- 🚀 Start the FastAPI server
- 🌐 Launch the web interface at `http://localhost:8501`
- 📊 Display model performance metrics
- 🔍 Enable real-time X-ray analysis

## Project Structure

```
.
├── app.py                    # FastAPI web application
├── train_eval_tf.py         # Training script
├── requirements.txt         # Dependencies
├── train.sh                # Training launcher
├── run.sh                  # Application launcher
├── models/                 # Saved models & metrics
│   ├── best_model.keras
│   ├── best_threshold.txt
│   └── training_metrics.json
└── templates/              # Web interface templates
    └── index.html
```

## Model Architecture

The system uses a fine-tuned DenseNet-121 architecture:
- Pre-trained on ImageNet
- Custom top layers for fracture detection
- Dropout and batch normalization for regularization
- L2 regularization on dense layers
- Gradient clipping for stable training

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- DenseNet-121 architecture by Huang et al.
- TensorFlow and FastAPI communities
- Contributors and maintainers

## Contact

Your Name - [@yourusername](https://twitter.com/yourusername)

Project Link: [https://github.com/yourusername/xray-fracture-detection](https://github.com/yourusername/xray-fracture-detection)