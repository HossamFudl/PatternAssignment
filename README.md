# 🐕🐱 Dogs vs Cats CNN Classifier

A deep learning image classifier that uses Convolutional Neural Networks (CNN) to distinguish between dogs and cats with high accuracy. Built with TensorFlow and Keras.

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Results](#results)

## ✨ Features

- **Automated Dataset Organization**: Automatically organizes mixed dog/cat images into separate folders
- **Data Augmentation**: Implements rotation, shifting, shearing, and flipping to improve model generalization
- **CNN Architecture**: 4-layer convolutional neural network with dropout regularization
- **Interactive Prediction**: Test your trained model on any image with an easy-to-use interface
- **Training Visualization**: Automatic plotting of accuracy and loss metrics
- **Model Persistence**: Save and load trained models for future use
- **Real-time Predictions**: Get instant predictions with confidence scores

## 🏗️ Architecture

The CNN model consists of:
- **4 Convolutional Blocks**: Each with Conv2D + MaxPooling layers (32, 64, 128, 128 filters)
- **Dropout Layer**: 0.5 dropout rate to prevent overfitting
- **Dense Layers**: 512-unit fully connected layer + sigmoid output
- **Total Parameters**: ~15M trainable parameters

```
Input (150x150x3) → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool 
→ Conv2D(128) → MaxPool → Conv2D(128) → MaxPool → Flatten 
→ Dropout(0.5) → Dense(512) → Dense(1, sigmoid)
```

## 📦 Requirements

### Python Version
- Python 3.7 or higher

### Dependencies
```
tensorflow>=2.8.0
numpy>=1.19.5
matplotlib>=3.3.4
Pillow>=8.0.0
```

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/HossamFudl/CatDogImageClassification.git
cd dogs-vs-cats-classifier
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file with:
```txt
tensorflow>=2.8.0
numpy>=1.19.5
matplotlib>=3.3.4
Pillow>=8.0.0
```

## 📁 Dataset Setup

### Download the Dataset

1. **Go to Kaggle**: [Dogs vs Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
2. **Download** the dataset (requires Kaggle account)
3. **Extract** the downloaded zip file

### Organize Your Files

After extraction, your directory structure should look like:

```
dogs-vs-cats-classifier/
│
├── train/                  # Place all training images here
│   ├── dog.0.jpg
│   ├── dog.1.jpg
│   ├── cat.0.jpg
│   ├── cat.1.jpg
│   └── ... (25,000 images total)
│
├── dogs_vs_cats_classifier.py
├── requirements.txt
└── README.md
```

**Important Notes:**
- The `train` folder should contain all 25,000 images from the dataset
- Image filenames must contain either `dog` or `cat` (e.g., `dog.123.jpg`, `cat.456.jpg`)
- The script will automatically create a `train_organized/` folder with subdirectories for dogs and cats
- Supported formats: `.jpg`, `.jpeg`, `.png`

### Dataset Statistics
- **Total Images**: 25,000
- **Dogs**: 12,500 images
- **Cats**: 12,500 images
- **Image Format**: JPEG
- **Training Split**: 80% training, 20% validation (automatic)

## 💻 Usage

### Training a New Model

Simply run the script:
```bash
python dogs_vs_cats_classifier.py
```

The script will:
1. ✅ Organize your dataset automatically
2. ✅ Create and compile the CNN model
3. ✅ Train for 25 epochs with data augmentation
4. ✅ Save the model as `dogs_vs_cats_model.h5`
5. ✅ Generate training history plots
6. ✅ Offer interactive prediction mode

### Loading an Existing Model

If a trained model exists, you'll see:
```
⚠️  Found existing model: dogs_vs_cats_model.h5
Do you want to (L)oad existing model, (R)etrain, or (Q)uit? [L/R/Q]:
```

- **L**: Load and use existing model
- **R**: Retrain from scratch
- **Q**: Quit the program

### Making Predictions

In interactive prediction mode:
```
📁 Enter image path: path/to/your/image.jpg
🔍 Analyzing image...

✅ Prediction: 🐕 DOG
📊 Confidence: 94.32%
💾 Result saved as: prediction_result.png
```

### Configuration Options

Edit these variables at the top of the script:
```python
IMG_SIZE = 150        # Image dimensions (150x150 pixels)
BATCH_SIZE = 32       # Batch size for training
EPOCHS = 25           # Number of training epochs
```

## 📊 Model Performance

Expected performance metrics:
- **Training Accuracy**: ~90-95%
- **Validation Accuracy**: ~85-90%
- **Training Time**: ~1-2 hours on CPU, ~15-20 minutes on GPU
- **Model Size**: ~180 MB

### Training Output Example
```
Epoch 25/25
625/625 [==============================] - 45s 72ms/step 
loss: 0.1234 - accuracy: 0.9456 - val_loss: 0.2345 - val_accuracy: 0.8923

📊 Final Results:
   Training Accuracy: 94.56%
   Validation Accuracy: 89.23%
```

## 📂 Project Structure

```
dogs-vs-cats-classifier/
│
├── train/                      # Original dataset images
├── train_organized/            # Auto-generated organized dataset
│   ├── dogs/                   # Dog images
│   └── cats/                   # Cat images
│
├── dogs_vs_cats_classifier.py  # Main script
├── dogs_vs_cats_model.h5       # Saved trained model
├── training_history.png        # Training metrics plot
├── prediction_result.png       # Latest prediction visualization
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔬 How It Works

### 1. Data Preprocessing
- Images resized to 150x150 pixels
- Pixel values normalized to [0, 1] range
- Data augmentation applied during training

### 2. Data Augmentation
Training images undergo random transformations:
- Rotation: ±40 degrees
- Width/Height shifts: ±20%
- Shear transformation: ±20%
- Zoom: ±20%
- Horizontal flip

### 3. Model Training
- **Optimizer**: Adam
- **Loss Function**: Binary crossentropy
- **Metrics**: Accuracy
- **Regularization**: Dropout (0.5)

### 4. Prediction
- Load image → Resize → Normalize → Predict
- Output: Class (Dog/Cat) + Confidence score

## 📈 Results

The model generates two outputs:

### 1. Training History Plot (`training_history.png`)
Shows training/validation accuracy and loss over epochs

### 2. Prediction Results (`prediction_result.png`)
Visual display of prediction with confidence score

## 🛠️ Troubleshooting

### Common Issues

**Issue**: "No images found or unable to organize dataset"
- **Solution**: Ensure image filenames contain `dog` or `cat`
- Check that images are in the `train/` folder

**Issue**: "Out of memory" error
- **Solution**: Reduce `BATCH_SIZE` (try 16 or 8)
- Use a machine with more RAM or enable GPU

**Issue**: Low accuracy (< 70%)
- **Solution**: Train for more epochs (try 30-50)
- Ensure dataset is properly organized
- Check that images aren't corrupted

**Issue**: Training is very slow
- **Solution**: Use a GPU-enabled environment
- Consider using a smaller subset for testing
- Reduce `EPOCHS` or `IMG_SIZE` for faster training



## 🙏 Acknowledgments

- Dataset: [Kaggle Dogs vs Cats Competition](https://www.kaggle.com/c/dogs-vs-cats)




⭐ **If you found this project helpful, please give it a star!** ⭐
