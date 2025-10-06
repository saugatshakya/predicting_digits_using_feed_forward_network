````markdown
# MNIST Digit Classification using Feedforward Neural Network

This project implements a **feedforward neural network (multilayer perceptron, MLP)** in PyTorch to classify handwritten digits from the MNIST dataset. It also includes functionality to test the model on custom JPG images.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Testing on MNIST](#testing-on-mnist)
- [Testing on JPG Images](#testing-on-jpg-images)
- [Evaluation Metrics](#evaluation-metrics)
- [References](#references)

---

## Project Overview

- Dataset: MNIST handwritten digits (0–9)
- Image size: 28×28 pixels, grayscale
- Model type: Feedforward Neural Network (MLP)
- Purpose: Classify digits and evaluate performance using accuracy, precision, recall, and F1 score

---

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- PIL (Pillow)
- scikit-learn

Install dependencies:

```bash
pip install torch torchvision numpy pillow scikit-learn
```
````

---

## Data Preparation

### MNIST Dataset

- Download MNIST raw files (`train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, etc.) from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/).
- Load and normalize images to `[0,1]`.
- Flatten 28×28 images into 784-element vectors.
- Use `TensorDataset` and `DataLoader` to create training batches.

### Custom JPG Images

- Convert to grayscale
- Resize to 28×28 pixels
- Normalize pixel values to `[0,1]`
- Flatten to 1D vector
- Convert to PyTorch tensor

---

## Model Architecture

```python
nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
```

- **Input layer**: 784 neurons (28×28 pixels)
- **Hidden layers**: 128 → 64 neurons, with ReLU activations
- **Output layer**: 10 neurons (digits 0–9)

---

## Training

- Loss function: `CrossEntropyLoss`
- Optimizer: Adam, learning rate 0.001
- Epochs: 15
- Device: CPU or GPU (`cuda` if available)

Training loop:

1. Forward pass: compute logits
2. Compute loss
3. Backpropagation: `loss.backward()`
4. Update weights: `optimizer.step()`
5. Track loss per epoch

---

## Testing on MNIST

- Set model to evaluation mode: `model.eval()`
- Disable gradient computation: `torch.no_grad()`
- Use `DataLoader` for test dataset
- Collect predictions and compute metrics using scikit-learn:

```python
from sklearn.metrics import classification_report, f1_score
```

---

## Testing on JPG Images

1. Prepare image:

```python
from PIL import Image
import numpy as np
import torch

def prepare_image(image_path, device="cpu"):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_flat = img_array.flatten()
    img_tensor = torch.tensor(img_flat, dtype=torch.float32).unsqueeze(0).to(device)
    return img_tensor
```

2. Predict:

```python
model.eval()
img_tensor = prepare_image("test_digit.jpg", device=device)
with torch.no_grad():
    output = model(img_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
print(f"Predicted digit: {predicted_class}")
```

- Supports single or batch JPG images

---

## Evaluation Metrics

- **Accuracy**
- **Precision, Recall, F1-score** (per class)
- **Macro F1-score**

---

## References

- Official MNIST description by Yann LeCun: [https://yann.lecun.org/exdb/mnist/index.html](https://yann.lecun.org/exdb/mnist/index.html)

---
