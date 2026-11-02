# 🧠 CNN Image Classification — CIFAR-10

A complete deep-learning image classification project built with **PyTorch**, covering CNN fundamentals, regularization, normalization, learning-rate scheduling, residual networks, error analysis, and transfer learning with a pretrained ResNet18.

This project was developed as part of my **AI/ML → AI Security learning roadmap**, with an emphasis on understanding *why* each technique works rather than treating neural networks as black boxes.

---

## 📌 Project Overview

The goal of this project is to build and progressively improve image-classification models on the **CIFAR-10** dataset while understanding the underlying deep-learning concepts.

The project progresses from a simple CNN trained from scratch to a deeper residual network and finally to **transfer learning with a pretrained ResNet18**.

### Main progression

```text
Basic CNN
   ↓
Data Augmentation
   ↓
Batch Normalization
   ↓
Checkpointing
   ↓
Learning-Rate Scheduling
   ↓
Residual Networks
   ↓
Global Average Pooling
   ↓
Transfer Learning
   ↓
Feature Extraction
   ↓
Fine-Tuning
```

The final fine-tuned ResNet18 achieved:

> **76.02% test accuracy on CIFAR-10**

---

# 📊 Dataset — CIFAR-10

CIFAR-10 contains:

- **60,000 color images**
- **32 × 32 pixels**
- **3 RGB channels**
- **10 classes**
- 50,000 training images
- 10,000 test images

The 10 classes are:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

Each class contains 6,000 images in total.

For this project, the original training set was split into:

```text
45,000 → Training
5,000  → Validation
10,000 → Test
```

The train/validation split used a fixed random seed for reproducibility.

---

# 🧠 What I Learned

## 1. CNN Fundamentals

The project started by building CNNs from first principles.

Topics covered:

- Convolution
- Kernels / filters
- Feature maps
- Input channels
- Output channels
- RGB images
- Stride
- Padding
- Pooling
- Flattening
- Fully connected layers

### Tensor shape reasoning

A typical CIFAR-10 batch:

```text
[batch, channels, height, width]

[64, 3, 32, 32]
```

The project repeatedly tracked tensor shapes through the network to understand how spatial dimensions and channels change.

---

# 🔬 Convolution Concepts

I manually explored convolution before relying on `nn.Conv2d`.

Topics included:

- Manual convolution
- Multi-channel convolution
- RGB filtering
- `F.conv2d`
- `nn.Conv2d`
- Parameter counting
- Output-size calculations

For example:

```python
nn.Conv2d(3, 4, kernel_size=3)
```

contains:

```text
4 filters
× 3 input channels
× 3 × 3 kernel
+ 4 biases

= 112 parameters
```

This helped establish the relationship between convolution architecture and learnable parameters.

---

# 🏗️ Baseline CNN

The first real CIFAR-10 model was:

```python
baseline_cnn = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Flatten(),
    nn.Linear(64 * 8 * 8, 10)
)
```

Parameter count:

```text
60,362
```

### Baseline result

**Test accuracy: 53.04%**

This provided the reference point for later improvements.

---

# 🧪 Data Augmentation

I experimented with image augmentation using:

- Random horizontal flipping
- Random cropping
- Padding

Example:

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()
])
```

The purpose was to expose the model to slightly different versions of the same image and improve generalization.

The augmentation experiment did not improve the model within the 10-epoch experiment, reaching approximately **47.06% validation accuracy**.

This reinforced an important lesson:

> Adding augmentation does not automatically improve accuracy. Its usefulness depends on the dataset, transformation choices, model, optimizer, and training setup.

---

# 📐 Batch Normalization

Batch Normalization was then added after convolution layers.

```python
nn.Conv2d(...)
nn.BatchNorm2d(...)
nn.ReLU()
```

The CNN with BatchNorm had:

```text
60,554 parameters
```

compared with:

```text
60,362 parameters
```

for the baseline.

I also investigated the difference between:

```python
model.train()
```

and:

```python
model.eval()
```

for BatchNorm.

This demonstrated that BatchNorm behaves differently during training and inference because training uses batch statistics while evaluation uses running statistics.

---

# 💾 Checkpointing

Instead of automatically using the final epoch, the model was checkpointed whenever validation accuracy improved.

Conceptually:

```text
Train
  ↓
Validation
  ↓
Is validation accuracy better?
  ↓
Yes → Save model
```

The best validation checkpoint was restored before testing.

This established an important ML methodology:

> **Validation selects the model. Test evaluates the selected model.**

---

# 📉 Learning-Rate Scheduling

I introduced `StepLR` to reduce the learning rate during training.

```python
StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)
```

The learning rate progressed approximately as:

```text
Epochs 1–3  → 0.01
Epochs 4–6  → 0.001
Epochs 7–9  → 0.0001
Epoch 10    → 0.00001
```

The best checkpointed CNN reached:

**69.54% validation accuracy**

and:

**69.46% test accuracy**

This was a major improvement over the original 53.04% baseline.

---

# 🔍 CNN Error Analysis

Rather than stopping at accuracy, I analyzed model behavior.

## Confusion Matrix

The final CNN was evaluated using a confusion matrix to identify which classes were frequently confused.

Important confusion patterns included:

- Cat ↔ Dog
- Automobile ↔ Truck
- Bird → Deer / Dog / Frog / Airplane
- Ship → Airplane

This showed that overall accuracy alone does not explain *where* the model struggles.

---

# 🎯 Per-Class Accuracy

The final CNN achieved approximately:

| Class | Accuracy |
|---|---:|
| Airplane | 72.1% |
| Automobile | 79.6% |
| Bird | 53.6% |
| Cat | 52.0% |
| Deer | 65.7% |
| Dog | 62.4% |
| Frog | 78.7% |
| Horse | 74.0% |
| Ship | 78.5% |
| Truck | 78.0% |

The weakest classes were primarily visually similar categories such as:

```text
cat / dog
bird / other animals
automobile / truck
```

---

# ⚠️ Confidence Analysis

I also investigated cases where the model was **very confident but wrong**.

Examples included predictions such as:

```text
automobile → truck
ship → automobile
bird → airplane
cat → truck
deer → horse
```

Some incorrect predictions had confidence above **99%**.

This demonstrated an important distinction:

> **Softmax confidence is not the same thing as probability that the prediction is actually correct.**

A neural network can be extremely confident and still be wrong.

---

# 🧠 Receptive Fields

The project also explored receptive fields.

A neuron's receptive field represents the region of the original input that can influence its activation.

For example, stacking convolution and pooling layers progressively increases the effective receptive field.

This led to understanding why deeper CNNs can capture increasingly large-scale patterns.

---

# 🔗 Residual Networks

The project then moved beyond the basic CNN architecture.

We studied the problem of:

- Vanishing gradients
- Degradation in deeper networks
- Difficulty training deeper architectures

This introduced the key ResNet idea:

```text
Input
  │
  ├──────────────┐
  │              │
  ▼              │
Layers           │
  │              │
  ▼              │
F(x)             │
  │              │
  └────── + <────┘
         │
         ▼
       F(x)+x
```

Instead of forcing the layers to learn the entire transformation, residual blocks learn a residual function while the shortcut provides a direct path.

---

# 🧩 Projection Shortcuts

When the input and output dimensions do not match, a projection shortcut can be used.

For example:

```text
32 channels
   ↓
1×1 convolution
   ↓
64 channels
```

A stride can also be used to perform spatial downsampling.

This allowed the implementation of residual blocks that change both:

- number of channels
- spatial resolution

---

# 🌐 Global Average Pooling

I also explored Global Average Pooling using:

```python
nn.AdaptiveAvgPool2d((1, 1))
```

For example:

```text
[1, 128, 8, 8]
        ↓
[1, 128, 1, 1]
        ↓
[1, 128]
```

This significantly reduces the number of parameters in the final classifier compared with flattening the entire feature map.

---

# 🏗️ Mini-ResNet

A custom Mini-ResNet was implemented using:

```text
Input
 ↓
Conv + BatchNorm + ReLU
 ↓
Residual Block
 ↓
Residual Block
 ↓
Downsampling Residual Block
 ↓
Residual Block
 ↓
Downsampling Residual Block
 ↓
Residual Block
 ↓
Global Average Pooling
 ↓
Linear
 ↓
10 classes
```

Parameter count:

```text
697,738
```

### Result

**Test accuracy: 72.60%**

This improved upon the previous best CNN result of 69.46%.

However, the improvement cannot be attributed solely to residual connections because the architecture also changed in depth, width, BatchNorm usage, and classifier design.

---

# 🚀 Transfer Learning

The final major section introduced **transfer learning**.

Instead of training an image model entirely from scratch, I used a model pretrained on ImageNet:

```python
model = torchvision.models.resnet18(weights="DEFAULT")
```

The pretrained ResNet18 contains approximately:

```text
11.69 million parameters
```

and originally produces predictions for:

```text
1000 ImageNet classes
```

---

# 🔄 Adapting ResNet18 to CIFAR-10

The original classifier was replaced:

```python
model.fc = nn.Linear(
    model.fc.in_features,
    10
)
```

This changed:

```text
512 → 1000
```

into:

```text
512 → 10
```

for CIFAR-10 classification.

---

# 🧊 Feature Extraction

First, the pretrained backbone was frozen:

```python
for param in model.parameters():
    param.requires_grad = False
```

Only the new classifier was trained.

Trainable parameters:

```text
5,130
```

The classifier was trained for 5 epochs.

### Best validation result

**46.20%**

This demonstrated an important limitation:

> Pretrained features are useful, but features learned for ImageNet are not automatically optimal for a different dataset.

---

# 🔥 Fine-Tuning

The next step was to allow deeper ResNet layers to adapt to CIFAR-10.

The following components were made trainable:

```text
layer3
layer4
fc
```

while earlier layers remained frozen.

Approximately:

```text
10.5 million
```

parameters became trainable.

A smaller learning rate was used:

```text
0.0001
```

because pretrained weights should generally be updated more carefully than a randomly initialized classifier.

---

# 🏆 Final Transfer-Learning Result

The best fine-tuned checkpoint achieved:

```text
Best validation accuracy: 75.94%
Best epoch: 5
```

After restoring the best validation checkpoint:

```text
Test accuracy: 76.02%
Test loss:     0.8448
```

Correct predictions:

```text
7,602 / 10,000
```

Incorrect predictions:

```text
2,398 / 10,000
```

The validation and test results were extremely close:

```text
Validation: 75.94%
Test:       76.02%
Difference: 0.08 percentage points
```

---

# 📈 Model Progression

The major results were:

| Model | Result |
|---|---:|
| Basic CNN | **53.04% test** |
| CNN + BatchNorm + StepLR | **69.46% test** |
| Mini-ResNet | **72.60% test** |
| ResNet18 Feature Extraction | **46.20% validation** |
| ResNet18 Fine-Tuning | **76.02% test** |

### Overall improvement

Compared with the original CNN:

```text
53.04%
   ↓
76.02%
```

Improvement:

**+22.98 percentage points**

---

# 🧪 Experimental Lessons

This project was not just about maximizing accuracy.

Several important lessons came from the experiments.

### 1. More complexity does not automatically mean better results

A model can be deeper and still perform poorly if the training setup is inappropriate.

### 2. Validation and test sets have different jobs

```text
Training → learn
Validation → choose
Test → final evaluation
```

### 3. Checkpointing matters

The final epoch is not necessarily the best model.

### 4. BatchNorm changes training and inference behavior

`train()` and `eval()` must be used correctly.

### 5. Augmentation is not guaranteed to improve performance

The transformation must match the problem.

### 6. Confidence can be misleading

A model can be highly confident and completely wrong.

### 7. Error analysis reveals more than accuracy

A confusion matrix can expose specific weaknesses hidden by overall accuracy.

### 8. Transfer learning can dramatically improve performance

A pretrained model can provide much stronger representations than a small CNN trained from scratch.

### 9. Fine-tuning matters

Frozen ImageNet features achieved only 46.20% validation accuracy, while fine-tuning deeper layers increased performance to 75.94% validation / 76.02% test.

---

# 🔐 Connection to AI Security

This project also establishes foundations for future **AI Security** work.

Understanding how image classifiers behave is important before studying attacks against them.

The project already explored concepts relevant to model security:

- model confidence
- high-confidence incorrect predictions
- generalization
- robustness
- distribution changes
- model architecture
- transfer learning
- model behavior under different inputs

These concepts will later connect to:

```text
CNN
 ↓
Adversarial Examples
 ↓
Evasion Attacks
 ↓
Adversarial Training
 ↓
Robustness
 ↓
AI Security
```

The future Deep Learning Security project will build on these concepts rather than treating security as a completely separate subject.

---

# 🛠️ Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- CUDA
- Jupyter Notebook
- CIFAR-10
- ResNet18
- ImageNet pretrained weights

---

# 📁 Project Structure

```text
cnn-cifar-10/
│
├── cnn-cifar-10.ipynb
├── README.md
└── .gitignore
```

The CIFAR-10 dataset itself is intentionally **not included in the repository**.

---

# ▶️ Running the Notebook

The notebook was developed in **Kaggle** using GPU acceleration.

To reproduce the project:

1. Open the notebook in Kaggle.
2. Enable GPU acceleration.
3. Run the notebook from top to bottom.
4. Allow torchvision to download CIFAR-10 if necessary.
5. Allow torchvision to download pretrained ResNet18 weights when reaching the transfer-learning section.

The project can also be adapted to another PyTorch environment with CUDA support.

---

# 🎯 Final Takeaways

This project progressed from understanding a single convolution operation to training and analyzing modern pretrained deep-learning architectures.

The most important progression was:

```text
Understand convolution
        ↓
Understand CNN architecture
        ↓
Train a real image classifier
        ↓
Improve training
        ↓
Analyze errors
        ↓
Understand receptive fields
        ↓
Understand residual connections
        ↓
Build a ResNet
        ↓
Understand transfer learning
        ↓
Feature extraction
        ↓
Fine-tuning
```

The final result was a **76.02% CIFAR-10 test accuracy using a fine-tuned pretrained ResNet18**.

More importantly, the project established the deep-learning foundations required for the next stage of the roadmap:

```text
CNNs ✅
   ↓
RNNs
   ↓
LSTMs
   ↓
GRUs
   ↓
Attention
   ↓
Transformers
   ↓
🔐 Deep Learning Security Project
   ↓
NLP
   ↓
LLMs
   ↓
AI Engineering
   ↓
AI Security
```

---

## 📚 Learning Status

**CNN / Computer Vision Chapter: COMPLETE ✅**

Concepts covered:

- Convolution
- Channels
- RGB
- Conv2d
- Stride
- Padding
- Pooling
- Flattening
- Linear layers
- Logits
- Softmax
- Cross-Entropy
- Backpropagation
- DataLoader
- CNN training
- Data augmentation
- Dropout
- BatchNorm
- Checkpointing
- Learning-rate scheduling
- Receptive fields
- Residual networks
- Projection shortcuts
- Global Average Pooling
- Mini-ResNet
- Transfer learning
- Feature extraction
- Fine-tuning
- Error analysis
- Confidence analysis
