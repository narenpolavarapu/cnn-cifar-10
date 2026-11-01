# 🧠 CIFAR-10 CNN — From Convolution Basics to Real-World Error Analysis

> **A hands-on PyTorch CNN project exploring image classification, BatchNorm, learning-rate scheduling, checkpointing, augmentation, and model error analysis — built as part of my journey toward AI Engineering and AI Security.**

---

## 🚀 Project Overview

This project started with a simple question:

**How does a Convolutional Neural Network actually learn to recognize images?**

Instead of jumping directly into a high-performance architecture, I built the pipeline progressively and investigated what happens when different components are introduced.

The project uses **CIFAR-10**, a real-world image classification dataset containing 60,000 RGB images across 10 classes.

The progression was:

```text
CNN Fundamentals
      ↓
Baseline CNN
      ↓
Data Augmentation
      ↓
Batch Normalization
      ↓
Validation + Checkpointing
      ↓
Learning-Rate Scheduling
      ↓
Best Model Selection
      ↓
Confusion Matrix
      ↓
Visual Error Analysis
      ↓
Confidence Analysis
```

The final model achieved:

> 🏆 **69.46% test accuracy**

More importantly, the project was about understanding **why** the model behaves the way it does, rather than simply maximizing an accuracy number.

---

## 🎯 Goals

The main goals of this project were to understand:

- How convolutional layers process images
- How channels and feature maps work
- How pooling changes spatial dimensions
- How CNN outputs become class predictions
- How Cross-Entropy Loss works with logits
- How backpropagation updates convolutional filters
- How DataLoaders feed batches into a model
- How Batch Normalization affects optimization
- How Dropout affects training
- How data augmentation affects generalization
- Why validation and test sets must be separated
- How checkpointing prevents us from keeping a bad final epoch
- How learning-rate scheduling affects optimization
- How to interpret confusion matrices
- How to investigate individual model failures
- Why confidence does not necessarily mean correctness

---

# 📦 Dataset — CIFAR-10

CIFAR-10 contains:

- **60,000** color images
- **50,000** training images
- **10,000** test images
- Image size: **32 × 32**
- Channels: **3 (RGB)**
- Classes: **10**

### Classes

| Label | Class |
|---:|---|
| 0 | ✈️ Airplane |
| 1 | 🚗 Automobile |
| 2 | 🐦 Bird |
| 3 | 🐱 Cat |
| 4 | 🦌 Deer |
| 5 | 🐶 Dog |
| 6 | 🐸 Frog |
| 7 | 🐴 Horse |
| 8 | 🚢 Ship |
| 9 | 🚚 Truck |

Each class contains 6,000 images, making CIFAR-10 balanced.

---

# 🧪 Data Pipeline

The original 50,000 training images were split into:

```text
Training:   45,000
Validation:  5,000
Test:       10,000
```

The train/validation split was made reproducible using:

```python
torch.Generator().manual_seed(42)
```

The test set remained completely separate.

### Tensor shape

A batch enters the CNN as:

```text
[Batch, Channels, Height, Width]

[64, 3, 32, 32]
```

where:

- `64` = batch size
- `3` = RGB channels
- `32 × 32` = image dimensions

---

# 🧠 CNN Architecture

The baseline architecture was intentionally kept small:

```text
Input
  │
  ▼
Conv2d(3 → 32)
  │
ReLU
  │
MaxPool
  │
Conv2d(32 → 64)
  │
ReLU
  │
MaxPool
  │
Flatten
  │
Linear
  │
10 class logits
```

Implemented as:

```python
nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Flatten(),
    nn.Linear(64 * 8 * 8, 10)
)
```

### Shape flow

```text
[64, 3, 32, 32]
        ↓
[64, 32, 32, 32]
        ↓
[64, 32, 16, 16]
        ↓
[64, 64, 16, 16]
        ↓
[64, 64, 8, 8]
        ↓
[64, 4096]
        ↓
[64, 10]
```

Total parameters:

**60,362**

---

# 📊 Experiment 1 — Baseline CNN

The first real experiment established a baseline.

### Training

The model was trained for 10 epochs using:

- CrossEntropyLoss
- SGD
- Learning rate: `0.01`
- No BatchNorm
- No learning-rate scheduler
- No checkpoint restoration

### Result

```text
Test Accuracy: 53.04%
Test Loss:     1.2872
```

This became our reference point.

---

# 🔄 Experiment 2 — Data Augmentation

To improve robustness to small spatial changes, the training data was augmented using:

```python
transforms.RandomHorizontalFlip()
transforms.RandomCrop(32, padding=4)
```

The important idea here was:

> The model should not memorize one exact pixel arrangement of an object.

However, the augmentation experiment did **not** automatically improve the model.

Best validation accuracy:

```text
47.06%
```

This was an important lesson:

**A technique that often helps in deep learning is not guaranteed to help every model, dataset, or training configuration.**

---

# 🧮 Experiment 3 — Batch Normalization

Next, BatchNorm was introduced after the convolutional layers.

```text
Conv
 ↓
BatchNorm
 ↓
ReLU
 ↓
MaxPool
```

The BatchNorm model contained:

**60,554 parameters**

Only 192 additional learnable parameters came from BatchNorm.

### What happened?

BatchNorm dramatically accelerated optimization.

The model reached:

```text
~46% validation accuracy
```

after the first epoch and continued improving.

However, one run became unstable later in training, demonstrating another important lesson:

> **Training behavior and evaluation behavior are not the same thing when BatchNorm is involved.**

The model uses batch statistics during training and running statistics during evaluation.

---

# 💾 Experiment 4 — Checkpointing

Instead of blindly using the model from epoch 10, the best validation model was saved whenever validation accuracy improved.

Conceptually:

```python
if val_acc > best_val_acc:
    best_val_acc = val_acc
    best_state = copy.deepcopy(model.state_dict())
```

This separates:

```text
Training
    ↓
Validation → Select best checkpoint
    ↓
Test → Evaluate selected model
```

This is important because the final epoch is **not necessarily the best model**.

---

# 📉 Experiment 5 — Learning-Rate Scheduling

The next improvement was a `StepLR` scheduler:

```python
StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)
```

The learning rate followed:

```text
Epoch 1–3    → 0.01000
Epoch 4–6    → 0.00100
Epoch 7–9    → 0.00010
Epoch 10     → 0.00001
```

The idea:

> Start with larger updates to learn quickly, then reduce the learning rate so the model can make smaller, more stable improvements.

The optimizer update happens before the scheduler step:

```text
loss.backward()
      ↓
optimizer.step()
      ↓
scheduler.step()
```

---

# 🏆 Final Model

The final selected model combined:

- CNN
- BatchNorm
- SGD
- StepLR
- Validation checkpointing

The best checkpoint occurred at:

```text
Epoch: 7
Validation Accuracy: 69.54%
```

After restoring that checkpoint:

```text
Test Accuracy: 69.46%
Test Loss:     0.8948
```

### Improvement over baseline

```text
Baseline:  53.04%
Final:     69.46%

Improvement: +16.42 percentage points
```

The validation/test gap was only:

```text
69.54% - 69.46% = 0.08 percentage points
```

That gave us a useful indication that the selected checkpoint generalized similarly to the held-out test set.

---

# 📈 Experiment Summary

| Experiment | Best/Final Validation | Test Accuracy |
|---|---:|---:|
| Baseline CNN | 53.46% | **53.04%** |
| Data Augmentation | 47.06% | — |
| BatchNorm + checkpoint | 63.90% | **63.83%** |
| BatchNorm + StepLR + checkpoint | **69.54%** | **69.46%** |

> Results are specific to this architecture, training configuration, random initialization, and 10-epoch training setup. They are not intended as a state-of-the-art CIFAR-10 benchmark.

---

# 🔍 Confusion Matrix

The final model produced:

```text
[[721, 20, 55, 22, 16,  6, 11,  9, 97, 43],
 [ 22,796,  8, 12,  1,  4,  7,  6, 42,102],
 [ 72,  5,536, 53,107, 84, 74, 38, 18, 13],
 [ 24, 10, 63,520, 75,178, 64, 27, 19, 20],
 [ 28,  4, 72, 62,657, 35, 55, 65, 18,  4],
 [ 18,  3, 49,169, 45,624, 22, 52, 12,  6],
 [  7,  2, 42, 74, 41, 25,787,  9,  8,  5],
 [ 18,  4, 34, 44, 65, 71,  9,740,  2, 13],
 [ 88, 45,  4, 13,  6,  8,  7,  5,785, 39],
 [ 39, 88,  7, 14,  7,  8,  9, 14, 34,780]]
```

Each row contains exactly 1,000 test examples.

---

# 🎯 Per-Class Performance

| Class | Accuracy |
|---|---:|
| ✈️ Airplane | 72.1% |
| 🚗 Automobile | 79.6% |
| 🐦 Bird | 53.6% |
| 🐱 Cat | 52.0% |
| 🦌 Deer | 65.7% |
| 🐶 Dog | 62.4% |
| 🐸 Frog | 78.7% |
| 🐴 Horse | 74.0% |
| 🚢 Ship | 78.5% |
| 🚚 Truck | 78.0% |

Overall:

**69.46%**

---

# 🧩 What Is the Model Struggling With?

The confusion matrix reveals much more than a single accuracy number.

### 🐱 Cat ↔ 🐶 Dog

```text
Cat → Dog: 178
Dog → Cat: 169
```

These errors were highly symmetric.

The model frequently encountered:

- blurry images
- unusual poses
- unclear silhouettes
- difficult backgrounds
- very small visual details

This suggests that the learned representation does not reliably separate some cat/dog examples.

---

### 🚗 Automobile → 🚚 Truck

```text
Automobile → Truck: 102
Truck → Automobile: 88
```

Again, there is substantial overlap.

At only **32×32 pixels**, shape, orientation, and background can make these classes difficult to distinguish.

---

### 🐦 Bird → Other Classes

Birds were one of the weaker classes:

```text
Bird accuracy: 53.6%
```

Common confusions included:

```text
Bird → Deer
Bird → Dog
Bird → Frog
Bird → Airplane
```

This shows that a model can perform well overall while still having significant weaknesses on specific classes.

---

# 😵 High-Confidence Wrong Predictions

One of the most interesting parts of the project was examining predictions where the model was **very confident but wrong**.

Some examples reached approximately:

```text
99%+
```

confidence while being incorrect.

The top examples included:

```text
Automobile → Truck    99.21%
Ship       → Automobile 99.06%
Bird       → Airplane  99.05%
Cat        → Truck     99.01%
Deer       → Horse     98.39%
```

This demonstrates an important ML concept:

> **Softmax confidence is not the same thing as correctness probability.**

A prediction with:

```text
softmax = 0.992
```

means the model assigned approximately 99.2% of its output probability mass to that class.

It does **not** guarantee that the prediction is actually correct.

---

# 📊 Confidence Analysis

Average confidence:

```text
Correct predictions: 73.67%
Wrong predictions:   51.56%
```

So the model was generally more confident when correct.

However:

```text
Highest confidence among wrong predictions: 99.21%
```

This is an important warning sign about model calibration and uncertainty.

A model can be:

```text
confident + correct
confident + wrong
uncertain + correct
uncertain + wrong
```

Understanding these cases becomes especially important in security-sensitive AI systems.

---

# 🧠 What I Learned

This project went beyond simply training a CNN.

### CNN fundamentals

I learned how:

```text
Images
  ↓
Convolution
  ↓
Feature Maps
  ↓
Activation
  ↓
Pooling
  ↓
Flattening
  ↓
Linear Layer
  ↓
Logits
  ↓
Prediction
```

works mathematically and in PyTorch.

### Training

I practiced:

- Forward propagation
- Loss calculation
- Backpropagation
- Gradient inspection
- Optimizer updates
- DataLoader batching
- Train/eval modes
- Checkpointing
- Learning-rate scheduling

### Generalization

I investigated:

- Data augmentation
- Validation vs test sets
- Overfitting behavior
- BatchNorm train/eval differences
- Model selection

### Model analysis

I went beyond accuracy and examined:

- Confusion matrices
- Per-class accuracy
- Specific error pairs
- Individual misclassified images
- Prediction confidence
- High-confidence failures

---

# 🔐 Why This Project Matters for AI Security

This project is also a foundation for my eventual transition into **AI Security**.

Before attacking an AI system, I need to understand how it behaves normally.

This project establishes that baseline.

I now have a model where I understand:

```text
Architecture
    ↓
Training
    ↓
Optimization
    ↓
Validation
    ↓
Predictions
    ↓
Errors
    ↓
Confidence
```

That creates the foundation for eventually asking security questions such as:

```text
Can an attacker deliberately change an image
without changing what a human sees?

            ↓

Can that cause the CNN to misclassify it?

            ↓

Can we generate such adversarial examples?

            ↓

How transferable are attacks?

            ↓

Can the model detect or defend against them?
```

Those questions belong to the next stage of the journey:

**Adversarial Machine Learning / AI Security.**

---

# 🛠️ Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook
- Kaggle

---

# ▶️ Running the Project

The main experiment is contained in:

```text
cnn-cifar-10.ipynb
```

The notebook contains the complete progression from dataset loading through final model evaluation and error analysis.

### Dataset

The CIFAR-10 dataset should be downloaded through `torchvision`.

The dataset itself is intentionally **not stored in this repository**.

This keeps the repository lightweight and reproducible.

---

# 📁 Repository Structure

```text
cnn-cifar-10/
│
├── README.md
└── cnn-cifar-10.ipynb
```

Dataset files and Kaggle-generated internal files are intentionally excluded.

---

# 📌 Key Takeaways

The biggest lesson from this project wasn't:

> "I got 69.46%."

It was:

> **I learned how to investigate why a neural network performs the way it does.**

A model is more than an accuracy number.

You need to understand:

```text
What did it learn?
Where does it fail?
How confident is it?
Does it generalize?
Which classes overlap?
What happens when training changes?
What happens when evaluation changes?
```

Those questions become even more important when the model eventually becomes the **target of an attacker**.

---

## 🚧 What's Next?

This project closes the first major part of my CNN journey.

Next:

```text
Receptive Fields
      ↓
1×1 Convolutions
      ↓
Deeper CNNs
      ↓
Optimization Challenges
      ↓
Skip Connections
      ↓
ResNet
      ↓
Bottleneck Architectures
      ↓
Transfer Learning
      ↓
CNN Mastery
```

Then the journey continues:

```text
CNN
 ↓
RNN
 ↓
LSTM
 ↓
Sequence Modeling
 ↓
Deep Learning Security Project
 ↓
NLP
 ↓
Attention
 ↓
Transformers
 ↓
LLMs
 ↓
RAG / Agents
 ↓
AI Engineering
 ↓
AI Security & Red Teaming
```

---

## 🔥 Status

**Completed — CNN / CIFAR-10 Phase**

**Best test accuracy: 69.46%**

**Next milestone: Deeper CNN architectures and ResNet fundamentals.**
