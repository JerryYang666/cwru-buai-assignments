# BUAI 446 – Homework 1: Reuters Newswire Classification (PyTorch Edition)

**Name:** Ruihuang Yang  
**NetID:** rxy216  
**Date:** October 31, 2025

---

## Overview

This project implements a **multiclass text classifier** for the Reuters newswire dataset using **PyTorch**. The goal is to classify news articles into one of 46 mutually-exclusive topics using a neural network built entirely with PyTorch's `nn.Module` framework.

Unlike Keras-based implementations, this project demonstrates:
- Manual training loop implementation
- PyTorch DataLoaders and TensorDatasets
- Native PyTorch loss functions and optimizers
- Model evaluation without compilation

---

## Dataset

**Reuters Newswire Dataset:**
- **Training samples:** 8,982 news articles
- **Test samples:** 2,246 news articles  
- **Vocabulary size:** 10,000 most frequent words
- **Number of classes:** 46 topics (mutually exclusive)
- **Input format:** Variable-length sequences of word indices
- **Target format:** Integer class labels (0-45)

The dataset is loaded via `tensorflow.keras.datasets.reuters` (for convenience only), but all modeling is done in PyTorch.

---

## Implementation Details

### Data Preprocessing
1. **Vectorization:** Convert variable-length word sequences to multi-hot encoded vectors (shape: `[n_samples, 10000]`)
2. **Labels:** Keep as integer class indices for `nn.CrossEntropyLoss` compatibility
3. **Validation split:** Hold out 1,000 samples from training data
4. **Batch processing:** DataLoader with batch size of 512

### Model Architecture
**MLP (Multilayer Perceptron):**
```python
ReutersMLP(
  Linear(10000 → 64)
  ReLU
  Linear(64 → 64)
  ReLU
  Linear(64 → 46)  # Logits output
)
```

**Configuration:**
- **Input dimension:** 10,000 features
- **Hidden layers:** 2 layers × 64 units each
- **Activation:** ReLU
- **Output:** 46 logits (no softmax in forward pass)
- **Loss function:** `nn.CrossEntropyLoss()` (combines LogSoftmax + NLLLoss)
- **Optimizer:** RMSprop with learning rate 1e-3
- **Training epochs:** 30

### Training Process
- Manual training loop with batch iteration
- Track training and validation accuracy per epoch
- Model evaluation on separate test set
- Device-agnostic code (CPU/CUDA)

---

## Questions Addressed

1. **How many samples are in the train and test sets?**
2. **How did you vectorize the inputs? Why is this appropriate?**
3. **Why keep labels as integers for `CrossEntropyLoss`? What changes with one-hot?**
4. **Model architecture definition and one improvement experiment**
5. **Training vs validation accuracy plot with overfitting analysis**
6. **Test accuracy and first sample prediction with interpretation**

---

## Project Structure

```
4-reuters-classification/
├── reuters-classification.py       # Main implementation (percent format)
├── reuters-classification.ipynb    # Jupyter notebook version
├── pyproject.toml                  # Dependencies (uv package manager)
├── uv.lock                         # Locked dependencies
├── README.md                       # This file
├── data/                           # Dataset cache (auto-downloaded)
└── graphs/                         # Visualizations output
```

---

## Setup & Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate  # Unix/macOS
# or
.venv\Scripts\activate     # Windows
```

### Required Dependencies
- `torch` – PyTorch framework
- `numpy` – Array operations
- `tensorflow` – Only for dataset loading (`keras.datasets.reuters`)
- `jupytext` – Sync between .py and .ipynb formats
- `matplotlib` – Visualization (for plots)

---

## File Format Conversion

This project uses **Jupytext** to maintain both `.py` and `.ipynb` versions.

### Convert .py to .ipynb
```bash
jupytext --to ipynb reuters-classification.py
```

### Convert .ipynb to .py
```bash
jupytext --to py:percent reuters-classification.ipynb
```

### Sync automatically
```bash
jupytext --set-formats py:percent,ipynb reuters-classification.ipynb
```

---

## Usage

### Run as Python script
```bash
python reuters-classification.py
```

### Run in Jupyter
```bash
jupyter notebook reuters-classification.ipynb
```

---

## Results Summary

- **Test Accuracy:** ~79% (exact value printed during execution)
- **Training Time:** ~30 epochs with progress printed every 5 epochs
- **Overfitting Analysis:** Compare training vs validation accuracy curves
- **First Sample Prediction:** Class label for `test_data[0]` is predicted and displayed

---

## Key Differences from Keras Implementation

| Aspect | Keras | PyTorch (This Project) |
|--------|-------|------------------------|
| Model definition | `Sequential()` | `nn.Module` class |
| Loss function | `categorical_crossentropy` | `nn.CrossEntropyLoss()` |
| Label format | One-hot encoded | Integer class indices |
| Training | `model.fit()` | Manual training loop |
| Evaluation | `model.evaluate()` | Manual evaluation loop |
| Predictions | `model.predict()` | `model(x)` in eval mode |

---

## Optional Extensions

- **Dropout:** Add `nn.Dropout()` layers to reduce overfitting
- **Weight decay:** Add L2 regularization via optimizer
- **Learning rate scheduling:** Use `torch.optim.lr_scheduler`
- **Deeper networks:** Experiment with more/wider hidden layers
- **Batch normalization:** Add `nn.BatchNorm1d()` layers

---

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Building Models with nn.Module](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [Training Loop Tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Reuters Dataset](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/reuters)

---

## Submission

**File name:** `LastName_FirstName_HW1.ipynb`  
**Format:** Jupyter notebook with all cells executed and outputs visible  
**Platform:** Canvas

Ensure all markdown explanations and visualizations are included before submission.
