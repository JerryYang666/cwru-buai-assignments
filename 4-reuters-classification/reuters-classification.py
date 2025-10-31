# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] id="cca12d22"
#
# # BUAI 446 – Fall 2024  
# ## Homework 1 (PyTorch Edition)
# Reuters Newswire Classification  
# **Name:** Ruihuang Yang  
# **NetID:** rxy216  
# **Date:** October 31, 2025  
#
# In this homework, you'll build and train a **multiclass text classifier** for the Reuters newswire dataset **using PyTorch**.
#
# We'll still use the same dataset (Reuters 46 topics) for consistency, but **all modeling, training, and evaluation must be done in PyTorch**.
#
# > Tip: In PyTorch you *don't compile a model*. Instead, you define a `nn.Module`, choose a loss (`nn.CrossEntropyLoss` for single‑label multiclass), pick an optimizer (e.g., `torch.optim.RMSprop`), and write a training loop that iterates over batches. See the official docs if needed.
#

# %% [markdown] id="93b7aefb"
#
# **Helpful references (optional):**  
# - Build models with `torch.nn` and `nn.Sequential` (PyTorch docs).  
# - Datasets & DataLoaders (PyTorch docs).  
# - Training loop basics (PyTorch tutorial).  
#
# *(Links included in the assignment PDF on Canvas.)*
#

# %% colab={"base_uri": "https://localhost:8080/"} id="94e8ba66" outputId="0fc2f218-caa9-49be-cd1b-8f2982eac7b5"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from tensorflow.keras.datasets import reuters  # used **only** to fetch the data
from typing import Tuple

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# %% colab={"base_uri": "https://localhost:8080/"} id="84591ad4" outputId="168e38b8-d9a1-49be-f94a-e35e65c4dfe5"

# Load Reuters dataset (10,000 most frequent words)
# This returns sequences of word indices with variable length, and integer labels in [0, 45].
(num_words, num_classes) = (10000, 46)
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=num_words)

len(train_data), len(test_data), len(train_labels), len(test_labels), max(train_labels), min(train_labels)


# %% colab={"base_uri": "https://localhost:8080/", "height": 122} id="5003c471" outputId="b4820b59-cd0f-484d-d9be-0f34c4edb610"

# See a decoded example to understand the data (optional exploratory cell)
word_index = reuters.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[9]])
decoded_newswire[:500]


# %% colab={"base_uri": "https://localhost:8080/"} id="7c2bc83d" outputId="a0b366d8-fe54-40f1-bb5c-4baff8889e9b"

def vectorize_sequences(sequences, dimension: int) -> np.ndarray:
    """Turns a list of sequences into a 2D numpy array of shape (len(sequences), dimension)
    where each row is a one‑hot multi-hot of word indices present in the sequence."""
    result = np.zeros((len(sequences), dimension), dtype=np.float32)
    for i, seq in enumerate(sequences):
        result[i, np.clip(seq, 0, dimension-1)] = 1.0
    return result

# 1) Vectorize inputs with one‑hot encoding (multi‑hot presence)
x_train = vectorize_sequences(train_data, num_words)
x_test = vectorize_sequences(test_data, num_words)

# 2) Labels: keep **integer class indices** for PyTorch CrossEntropyLoss
y_train = np.array(train_labels, dtype=np.int64)
y_test = np.array(test_labels, dtype=np.int64)

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# %% colab={"base_uri": "https://localhost:8080/"} id="6931acea" outputId="d0a36ef3-6333-428b-bf68-581470f1387e"

# Create tensors
X_train = torch.from_numpy(x_train)
y_train_t = torch.from_numpy(y_train)
X_test = torch.from_numpy(x_test)
y_test_t = torch.from_numpy(y_test)

# 3) Hold out 1,000 samples from training for validation
val_size = 1000
train_size = len(X_train) - val_size
train_ds, val_ds = random_split(TensorDataset(X_train, y_train_t), [train_size, val_size], generator=torch.Generator().manual_seed(42))

# DataLoaders
batch_size = 512
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)
test_loader = DataLoader(TensorDataset(X_test, y_test_t), batch_size=batch_size)
len(train_loader), len(val_loader), len(test_loader)


# %% colab={"base_uri": "https://localhost:8080/"} id="dcf5d232" outputId="47ab7050-8493-47a7-dc96-97d8faf6328d"

# 4) Define an MLP with two hidden layers (64 units each), ReLU, and a 46‑way output (logits)
class ReutersMLP(nn.Module):
    def __init__(self, in_dim=10000, hidden=64, num_classes=46):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)  # logits (no Softmax here)
        )
    def forward(self, x):
        return self.net(x)

model = ReutersMLP(in_dim=num_words, hidden=64, num_classes=num_classes).to(device)
model


# %% id="adf7a724"

# 5) Choose loss & optimizer
# For single‑label multiclass, use CrossEntropyLoss (expects raw logits + **integer** targets).
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)  # RMSprop to mirror the Keras spec

def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()



# %% id="616e85a3"

# 6-7) Train for 30 epochs, track train/val accuracy
epochs = 30
history = {"train_acc": [], "val_acc": []}

for epoch in range(1, epochs+1):
    model.train()
    running_acc = 0.0
    n_batches = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_acc += accuracy(logits.detach(), yb)
        n_batches += 1
    train_acc = running_acc / max(1, n_batches)

    # Validation
    model.eval()
    val_acc = 0.0
    n_val = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            val_acc += accuracy(logits, yb)
            n_val += 1
    val_acc = val_acc / max(1, n_val)

    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
        print(f"Epoch {epoch:02d} | train_acc={train_acc:.3f} | val_acc={val_acc:.3f}")

history


# %% id="920b6582"

# 8) Evaluate on the test set and predict the class for the first test sample
model.eval()
test_acc = 0.0
n = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        test_acc += accuracy(logits, yb)
        n += 1
test_acc = test_acc / max(1, n)
print("Test accuracy:", round(test_acc, 3))

# Predict the class of the first test sample
first_logits = model(torch.from_numpy(x_test[:1]).to(device))
pred_class = int(first_logits.argmax(dim=1).item())
pred_class


# %% [markdown] id="f0243615"
#
# ### What to submit
#
# Run all cells so outputs are visible, and submit your notebook as `LastName_FirstName_HW1.ipynb` on Canvas.
#
# **Answer these in your notebook (use Markdown cells where appropriate):**
# 1. How many samples are in the train and test sets?  
# 2. How did you vectorize the inputs? Explain why this is appropriate for this problem.  
# 3. Why do we keep labels as integer class indices for `CrossEntropyLoss`? What would change if you used one‑hot labels?  
# 4. Define your MLP as shown. Try **one improvement** (e.g., different hidden size, dropout, weight decay) and report the effect.  
# 5. Plot training vs validation accuracy across 30 epochs. Comment on overfitting/underfitting.  
# 6. Report final **test accuracy** and the predicted class for the first test sample. Briefly interpret the result.
#

# %% [markdown] id="f563e28c"
#
# > **Optional challenge (no extra credit):** Try adding `nn.Dropout` and describe the impact.  
#
#

# %% id="asMSlqwz5jIN"
