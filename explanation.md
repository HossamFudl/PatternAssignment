# Dogs vs Cats Classifier -- Code Explanation

This document explains the full Python code you provided for training a
**Convolutional Neural Network (CNN)** to classify dog and cat images
using TensorFlow/Keras.

------------------------------------------------------------------------

## 📌 1. Imports and Configuration

``` python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 25
```

-   Loads TensorFlow/Keras and other tools.
-   Defines the default image size, batch size, and number of training
    epochs.

------------------------------------------------------------------------

## 📌 2. Dataset Organization

### `organize_dataset()`

This function organizes images into:

    train_organized/
        ├── dogs/
        └── cats/

Steps: 1. Creates folders for dogs and cats. 2. Loops through image
files in the original `train` folder. 3. Copies images into the correct
folder based on filename (e.g., `"dog.12.jpg"`).

------------------------------------------------------------------------

## 📌 3. CNN Model Creation

### `create_cnn_model()`

Builds a standard CNN:

-   **4 convolutional + max-pooling layers**
-   **Flatten**
-   **Dropout (to reduce overfitting)**
-   **Dense layer with 512 units**
-   **Output layer** → 1 neuron + **sigmoid** → predicts Dog(1) or
    Cat(0)

------------------------------------------------------------------------

## 📌 4. Preparing Data

### `prepare_data()`

Uses **ImageDataGenerator**: - Augments images: rotate, zoom, flip,
shift. - Splits data (80% training, 20% validation). - Rescales all
images to values between 0--1.

Returns: - `train_generator` - `validation_generator`

------------------------------------------------------------------------

## 📌 5. Plotting Training History

### `plot_training_history(history)`

Creates two plots: 1. **Training vs Validation Accuracy** 2. **Training
vs Validation Loss**

Saves them as:

    training_history.png

------------------------------------------------------------------------

## 📌 6. Prediction on Single Image

### `predict_image(model, image_path)`

Steps: 1. Loads image and resizes to 150×150. 2. Converts to array and
normalizes. 3. Runs `model.predict`. 4. Displays image with prediction
label + confidence. 5. Saves result to:

    prediction_result.png

------------------------------------------------------------------------

## 📌 7. Interactive Prediction Mode

### `interactive_prediction(model)`

Allows typing image paths for real‑time prediction.

------------------------------------------------------------------------

## 📌 8. Main Function

### `main()`

This is the flow:

1.  Prints welcome banner.

2.  Checks if a saved model exists:

    -   Load it
    -   Retrain
    -   Or Quit

3.  Organizes dataset.

4.  Creates + compiles the CNN.

5.  Prepares data.

6.  Trains the model.

7.  Saves the model:

        dogs_vs_cats_model.h5

8.  Plots training history.

9.  Runs interactive test mode.

------------------------------------------------------------------------

## 📌 Summary

This script: ✔ Organizes dataset\
✔ Builds and trains a CNN\
✔ Evaluates training performance\
✔ Saves model\
✔ Lets you test images interactively

Perfect for beginners working with deep learning image classification.

------------------------------------------------------------------------
