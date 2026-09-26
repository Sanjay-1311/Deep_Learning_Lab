# Deep Learning Lab 6 — RNN, LSTM, GRU & Sequence Modeling

A comprehensive deep-learning laboratory implementation exploring **Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM), Gated Recurrent Units (GRUs), CNN-RNN architectures, and Encoder-Decoder sequence-to-sequence models**.

The lab uses both **sensor time-series data** and **video data** to study how recurrent architectures model temporal dependencies.

---

## 📌 Overview

This laboratory investigates sequential deep-learning models through four major experiments:

1. **Human Activity Recognition (HAR)**

   * Compare Vanilla RNN, LSTM, and GRU models.
   * Use inertial sensor time-series data from the UCI HAR Dataset.
   * Evaluate accuracy, precision, recall, F1-score, parameter count, and training time.

2. **Effect of Sequence Length**

   * Evaluate RNN, LSTM, and GRU using sequence lengths:

     * `T = 32`
     * `T = 64`
     * `T = 128`
   * Analyze the relationship between temporal context, performance, and training cost.

3. **Video Action Recognition**

   * Use a subset of UCF101.
   * Extract frame-level visual features using a frozen pretrained **MobileNetV2**.
   * Feed the resulting temporal feature sequences into:

     * CNN-LSTM
     * CNN-GRU

4. **Sequence-to-Sequence Learning**

   * Implement an Encoder-Decoder LSTM.
   * Train the model to reverse sequences of digits.
   * Demonstrate teacher forcing and autoregressive inference.

---

## 🧠 Learning Objectives

By completing this laboratory, the following concepts are demonstrated:

* Recurrent neural network fundamentals
* Hidden-state based temporal modeling
* Vanilla RNN limitations
* LSTM memory and gated computation
* GRU architecture
* Temporal sequence classification
* Sequence-length effects
* CNN-RNN hybrid architectures
* Transfer learning with MobileNetV2
* Frozen CNN feature extraction
* Video classification
* Encoder-Decoder architectures
* Teacher forcing
* Autoregressive decoding
* Token-level vs sequence-level accuracy
* Confusion matrices and per-class evaluation
* Model parameter and training-time comparison

---

# 🗂️ Experiments

## 1. Human Activity Recognition — RNN vs LSTM vs GRU

### Dataset

The first experiment uses the **UCI Human Activity Recognition Using Smartphones Dataset**.

The raw inertial signals contain:

* Body acceleration
* Gyroscope measurements
* Multiple sensor axes
* Temporal windows of sensor observations

The notebook loads the raw inertial signals and represents each sample as a temporal tensor:

```text
(samples, time steps, features)
```

The original sequence contains:

```text
128 time steps × 9 sensor features
```

### Dataset Preparation

A balanced subset is created using:

```text
400 samples per class
6 classes
----------------
2400 samples total
```

The data is divided using a stratified:

```text
70% Training
15% Validation
15% Testing
```

The input features are normalized using statistics calculated from the training set.

### Model Architecture

The same general architecture is used for all three recurrent models:

```text
Input
  │
  ▼
RNN / LSTM / GRU
32 recurrent units
  │
  ▼
Dense
16 units + ReLU
  │
  ▼
Dense
6 units + Softmax
  │
  ▼
Activity Class
```

### Training Configuration

| Parameter       | Value |
| --------------- | ----: |
| Recurrent units |    32 |
| Dense units     |    16 |
| Dropout         |   0.2 |
| Optimizer       |  Adam |
| Learning rate   | 0.001 |
| Batch size      |    32 |
| Epochs          |    30 |
| Random seed     |    42 |

The three models use the same training configuration to make the comparison more consistent.

---

## 📊 HAR Results

The final test-set results are:

| Model       | Accuracy | Precision | Recall |     F1 | Parameters | Training Time |
| ----------- | -------: | --------: | -----: | -----: | ---------: | ------------: |
| Vanilla RNN |   77.78% |    78.17% | 77.78% | 77.69% |      1,974 |       24.50 s |
| LSTM        |   94.17% |    94.38% | 94.17% | 94.17% |      6,006 |       19.81 s |
| GRU         |   95.00% |    95.01% | 95.00% | 95.00% |      4,758 |       19.76 s |

Precision, recall, and F1-score are **macro-averaged**.

The notebook also generates:

* Training/validation loss curves
* Training/validation accuracy curves
* Confusion matrices
* Per-class recall
* Model performance vs computational cost

---

# 2. Effect of Sequence Length

The second experiment investigates how changing the number of time steps affects recurrent-model performance.

The following sequence lengths are tested:

```text
T = 32
T = 64
T = 128
```

The same RNN, LSTM, and GRU architectures are trained for each sequence length.

### Test Macro F1

| Sequence Length |    RNN |   LSTM |    GRU |
| --------------: | -----: | -----: | -----: |
|              32 | 75.90% | 92.49% | 92.46% |
|              64 | 74.98% | 93.06% | 95.27% |
|             128 | 77.69% | 94.17% | 95.00% |

### Training Time

| Sequence Length |    RNN |   LSTM |    GRU |
| --------------: | -----: | -----: | -----: |
|              32 | 17.3 s | 15.7 s | 15.3 s |
|              64 | 18.8 s | 17.2 s | 16.9 s |
|             128 | 24.5 s | 19.8 s | 19.8 s |

The number of trainable parameters remains unchanged when sequence length changes:

```text
RNN  = 1,974
LSTM = 6,006
GRU  = 4,758
```

The notebook plots both:

* Sequence length vs test macro F1
* Sequence length vs training time

---

# 3. Video Action Recognition — CNN + RNN

The third experiment extends recurrent modeling to video.

Instead of directly feeding raw images into an RNN, the notebook uses a pretrained CNN to extract visual features from individual video frames.

### Dataset

A subset of **UCF101** is used containing five action classes:

```text
Basketball
Biking
TennisSwing
WalkingWithDog
JumpRope
```

A total of:

```text
300 videos
```

are downloaded.

Each class contains:

```text
30 training videos
15 validation videos
15 test videos
```

### Split Strategy

The videos are divided by their group identifier:

```text
Test:  groups g01–g05
Validation: g06–g10
Training: g11–g25
```

This prevents clips from closely related groups from being distributed across different splits.

---

## 🎞️ Frame Sampling

For every video:

```text
10 frames
```

are sampled uniformly across the entire video.

Each frame is:

```text
224 × 224 × 3
```

in RGB format.

Example transformation:

```text
Video
  │
  ├── Frame 1
  ├── Frame 2
  ├── ...
  └── Frame 10
```

---

# 4. MobileNetV2 Feature Extraction

A pretrained **MobileNetV2** is used as a frozen feature extractor.

The classification head is removed:

```python
MobileNetV2(
    include_top=False,
    weights="imagenet",
    pooling="avg"
)
```

The CNN is frozen:

```python
base_cnn.trainable = False
```

Each frame is converted into a:

```text
1280-dimensional feature vector
```

Therefore, every video becomes:

```text
10 frames × 1280 features
```

or:

```text
(10, 1280)
```

The complete dataset passed to the recurrent model has the shape:

```text
Training   : (150, 10, 1280)
Validation : (75, 10, 1280)
Testing    : (75, 10, 1280)
```

The frozen MobileNetV2 contains:

```text
2,257,984 parameters
```

These parameters are **not trainable**.

---

# 5. CNN-LSTM vs CNN-GRU

The extracted MobileNetV2 features are passed to recurrent layers.

### Architecture

```text
Video
  │
  ▼
10 uniformly sampled frames
  │
  ▼
MobileNetV2
Frozen CNN
  │
  ▼
10 × 1280 feature sequence
  │
  ├───────────────┐
  ▼               ▼
LSTM(32)        GRU(32)
  │               │
  ▼               ▼
Dropout(0.3)    Dropout(0.3)
  │               │
  ▼               ▼
Dense(5)        Dense(5)
  │               │
  ▼               ▼
Softmax         Softmax
```

### Training Configuration

| Parameter         | Value |
| ----------------- | ----: |
| Recurrent units   |    32 |
| Dropout           |   0.3 |
| Optimizer         |  Adam |
| Learning rate     | 0.001 |
| Batch size        |    16 |
| Epochs            |    30 |
| Frame count       |    10 |
| Feature dimension |  1280 |

The model selected as the main architecture is determined using validation performance.

---

## 📊 Video Classification Results

| Model    | Accuracy | Precision | Recall |     F1 | Trainable Parameters | Training Time |
| -------- | -------: | --------: | -----: | -----: | -------------------: | ------------: |
| CNN-LSTM |   85.33% |    86.33% | 85.33% | 85.30% |              168,229 |        6.95 s |
| CNN-GRU  |   80.00% |    80.02% | 80.00% | 79.94% |              126,309 |        6.71 s |

The notebook uses **CNN-LSTM as the main video model** based on validation selection.

The frozen MobileNetV2 parameters are not included in the trainable-parameter counts above.

---

# 6. Sequence-to-Sequence Learning

The final experiment implements an **Encoder-Decoder LSTM** for sequence reversal.

The model receives a sequence of six digits and must produce the sequence in reverse order.

Example:

```text
Input:
[3, 8, 1, 5, 2, 7]

Expected output:
[7, 2, 5, 1, 8, 3]
```

---

## Encoder

The encoder consists of:

```text
Input sequence
      │
      ▼
Embedding
16 dimensions
      │
      ▼
LSTM
128 units
      │
      ▼
Hidden state + Cell state
```

The final hidden state and cell state act as the context passed to the decoder.

---

## Decoder

The decoder receives:

```text
<START> + shifted target sequence
```

during training.

Architecture:

```text
Decoder input
      │
      ▼
Embedding
16 dimensions
      │
      ▼
LSTM
128 units
      │
      ▼
Dense
10 units + Softmax
```

---

## Teacher Forcing

During training, the decoder is provided with the correct previous output.

For example:

```text
Target:
[7, 2, 5, 1, 8, 3]

Decoder input:
[START, 7, 2, 5, 1, 8]
```

During actual inference, the decoder instead feeds its **own previous prediction** into the next step.

This demonstrates the difference between:

```text
Teacher-forced training
```

and:

```text
Autoregressive inference
```

---

## Seq2Seq Results

The model uses:

| Parameter           |  Value |
| ------------------- | -----: |
| Total sequences     | 10,000 |
| Sequence length     |      6 |
| Training split      |    70% |
| Validation split    |    15% |
| Test split          |    15% |
| Embedding dimension |     16 |
| LSTM units          |    128 |
| Epochs              |     40 |
| Batch size          |     64 |

Final results:

| Metric                |  Result |
| --------------------- | ------: |
| Token accuracy        | 100.00% |
| Sequence accuracy     | 100.00% |
| Final training loss   |  0.0004 |
| Final validation loss |  0.0006 |
| Parameters            | 150,090 |
| Training time         | 42.55 s |

Both token-level and complete sequence-level autoregressive accuracy reached:

```text
100%
```

---

# 📈 Consolidated Results

The complete set of major test results is summarized below.

| Model                | Dataset            |                 Accuracy | Macro F1 |
| -------------------- | ------------------ | -----------------------: | -------: |
| RNN                  | UCI HAR            |                   77.78% |   77.69% |
| LSTM                 | UCI HAR            |                   94.17% |   94.17% |
| GRU                  | UCI HAR            |                   95.00% |   95.00% |
| CNN-LSTM             | UCF101 subset      |                   85.33% |   85.30% |
| CNN-GRU              | UCF101 subset      |                   80.00% |   79.94% |
| Encoder-Decoder LSTM | Synthetic reversal | 100.00% token / sequence |  100.00% |

---

# 📊 Generated Visualizations

The notebook produces visualizations covering:

### UCI HAR

* Sensor signal vs time step
* RNN training/validation loss
* RNN training/validation accuracy
* LSTM training/validation curves
* GRU training/validation curves
* RNN vs LSTM vs GRU comparison
* Confusion matrices
* Per-class performance
* Sequence-length vs F1
* Sequence-length vs training time

### UCF101

* Uniformly sampled video frames
* CNN-LSTM/CNN-GRU training curves
* Video confusion matrices
* Per-action recall
* Example predictions and confidence

### Sequence-to-Sequence

* Training/validation loss
* Training/validation token accuracy
* Example input/expected/predicted sequences

---

# 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow
* Keras
* OpenCV
* MobileNetV2
* UCI HAR Dataset
* UCF101 Dataset

---

# 📁 Project Structure

A typical execution directory contains:

```text
.
├── lab-6-dl.ipynb
│
├── har_data/
│   └── UCI HAR Dataset/
│
├── ucf101_videos/
│   ├── *.avi
│   └── manifest.csv
│
├── video_features_mobilenetv2.npz
│
└── report_tables/
    ├── consolidated_results.csv
    ├── har_test_metrics.csv
    ├── f1_vs_sequence_length.csv
    ├── time_vs_sequence_length.csv
    ├── seq2seq_results.csv
    ├── har_per_class_recall.csv
    └── video_per_class_recall.csv
```

---

# 🚀 How to Run

## 1. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python
```

## 2. Open the notebook

```bash
jupyter notebook lab-6-dl.ipynb
```

or open it using:

* JupyterLab
* Google Colab
* Kaggle Notebooks

## 3. Run the notebook sequentially

The notebook is organized into blocks covering:

```text
Environment setup
        ↓
UCI HAR data
        ↓
RNN / LSTM / GRU
        ↓
Sequence-length experiment
        ↓
UCF101 video dataset
        ↓
Frame sampling
        ↓
MobileNetV2 features
        ↓
CNN-LSTM / CNN-GRU
        ↓
Seq2Seq Encoder-Decoder
        ↓
Consolidated results
```

---

# 💾 Dataset Requirements

The notebook downloads the **UCI HAR Dataset** automatically when possible.

For the video experiment, the notebook downloads the required UCF101 clips and creates a local manifest.

If automatic downloading is unavailable, the notebook contains fallback instructions for attaching the datasets manually.

Because the video experiment uses only a subset of UCF101, it does **not** require the complete UCF101 dataset.

---

# 🔬 Reproducibility

A fixed random seed is used throughout the experiments:

```python
SEED = 42
```

Random seeds are configured for:

* Python
* NumPy
* TensorFlow/Keras

The HAR experiment also uses stratified splitting, while the video experiment uses group-based splitting to avoid closely related clips crossing dataset boundaries.

---

# 🧩 Key Takeaways

### RNN vs LSTM vs GRU

The HAR experiment demonstrates that gated recurrent architectures can model the temporal sensor data more effectively than the basic Vanilla RNN used in this experiment.

The final HAR test results were:

```text
RNN  → 77.78% accuracy
LSTM → 94.17% accuracy
GRU  → 95.00% accuracy
```

### Sequence Length

Changing the sequence length affects both predictive performance and computational cost.

The experiment explicitly evaluates:

```text
32 → 64 → 128 time steps
```

while keeping the recurrent-layer parameter count unchanged.

### CNN + RNN

For video, the CNN handles the **spatial information** in individual frames, while the recurrent network handles the **temporal information** across frames.

```text
CNN → spatial features
RNN → temporal relationships
```

This produces the overall:

```text
CNN + LSTM / GRU
```

architecture.

### Encoder-Decoder

The sequence-to-sequence experiment demonstrates how an encoder can summarize an input sequence into a context representation and how a decoder can generate an output sequence step by step.

It also demonstrates the distinction between:

```text
Teacher forcing
```

and:

```text
Autoregressive decoding
```

---

# 👨‍💻 Author

**Deep Learning Laboratory — Lab 6**

B.Tech Artificial Intelligence & Data Science

---

## 📜 License

This project is intended for **educational and academic purposes**.
