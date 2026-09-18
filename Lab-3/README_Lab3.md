# CIFAR-10 CNN — Architecture Study & Feature Map Visualization

A convolutional neural network built from scratch (TensorFlow/Keras) for **CIFAR-10** image classification, paired with an in-depth visual analysis of what the network learns (filters, feature maps, pooling behavior) and a systematic ablation study over architectural and training hyperparameters.

## Overview

The notebook (`Lab-3-Dl.ipynb`) covers three main parts:

**1. Baseline CNN — build, train, evaluate**
1. Load CIFAR-10 via `tensorflow.keras.datasets`
2. Visualize sample images and the per-class distribution of the training set
3. Normalize pixel values to `[0, 1]`
4. Build a 3-block CNN: `(Conv32→Conv32→MaxPool) → (Conv64→MaxPool) → (Conv128→MaxPool) → Flatten → Dense(128) → Dense(10, softmax)`
5. Train for 15 epochs (Adam, sparse categorical cross-entropy, 10% validation split)
6. Plot training/validation accuracy and loss curves
7. Evaluate on the test set: test loss/accuracy, full classification report, confusion matrix, a grid of sample predictions (correct in green, incorrect in red)

**2. Interpretability — filters, feature maps, and pooling**
8. Extract and visualize intermediate Conv2D layer outputs (feature maps) for a sample test image
9. Compare feature maps from an early vs. a deep convolutional layer
10. Rebuild an equivalent model with `AveragePooling2D` in place of `MaxPooling2D` (reusing trained weights) and compare the two pooling strategies numerically (mean/max/std of activations) and visually
11. Inspect and print the parameter counts of each Conv2D layer, with a manual calculation verifying the parameter-count formula `(kernel_h × kernel_w × in_channels + 1) × filters`
12. Visualize the raw learned filters of the first convolutional layer
13. Recompute weighted accuracy/precision/recall/F1 with `scikit-learn`

**3. Ablation study — architectural & training hyperparameters**
14. Define a configurable `build_variant_model()` and a `train_and_eval()` helper (5 epochs each, for speed)
15. Sweep and compare test accuracy across: pooling type (max vs. average), kernel size (3×3 vs. 5×5), stride (1×1 vs. 2×2), padding (same vs. valid), number of filters ((16,32) / (32,64) / (64,128)), optimizer (Adam / SGD / RMSprop), and batch size (32 / 64 / 128)
16. Visualize each comparison as bar charts (including a zoomed-in view for pooling and kernel size)
17. Summarize all ablation results in a single pandas DataFrame

## Dataset

- **Dataset:** CIFAR-10 (`tensorflow.keras.datasets.cifar10`)
- **Classes (10):** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Size:** 50,000 training images / 10,000 test images, 32×32 RGB
- **Preprocessing:** pixel values scaled to `[0, 1]`; labels squeezed to 1D

## Model Architecture

**Baseline CNN:**

```
Input(32, 32, 3)
→ Conv2D(32, 3x3, ReLU, same) → Conv2D(32, 3x3, ReLU, same) → MaxPool(2x2)
→ Conv2D(64, 3x3, ReLU, same) → MaxPool(2x2)
→ Conv2D(128, 3x3, ReLU, same) → MaxPool(2x2)
→ Flatten → Dense(128, ReLU) → Dense(10, Softmax)
```

- Optimizer: Adam, Loss: sparse categorical cross-entropy
- Epochs: 15, Batch size: 64, Validation split: 10%

**Ablation variant model (`build_variant_model`):**

```
Input(32, 32, 3)
→ Conv2D(filters[0], kernel_size, stride, padding, ReLU) → Pooling(2x2)
→ Conv2D(filters[1], kernel_size, stride, padding, ReLU) → Pooling(2x2)
→ Flatten → Dense(128, ReLU) → Dense(10, Softmax)
```

Each ablation run trains for 5 epochs and reports test accuracy, isolating one dimension at a time:

| Dimension | Values compared |
|---|---|
| Pooling type | Max, Average |
| Kernel size | (3,3), (5,5) |
| Stride | (1,1), (2,2) |
| Padding | same, valid |
| Filters (Conv1, Conv2) | (16,32), (32,64), (64,128) |
| Optimizer | Adam, SGD, RMSprop |
| Batch size | 32, 64, 128 |

## Evaluation

The baseline model is evaluated on the test set with:

- Test loss and test accuracy
- Full per-class classification report (precision, recall, F1)
- Confusion matrix (annotated heatmap)
- Weighted accuracy/precision/recall/F1 (via `scikit-learn`)
- A grid of 25 random test predictions with true vs. predicted labels

Since exact metric values depend on the actual training run, they are produced by executing the notebook rather than fixed here.

## Interpretability Highlights

- **Feature maps:** intermediate outputs of every Conv2D layer are extracted and visualized for a chosen test image, contrasting what early layers vs. deep layers respond to.
- **Pooling comparison:** an equivalent model with AveragePooling2D (reusing the original trained Conv2D/Dense weights) is built to directly compare max vs. average pooling outputs on the same feature map, both numerically and visually.
- **Filter visualization:** the raw learned kernels of the first convolutional layer are normalized and plotted.
- **Parameter accounting:** per-layer Conv2D parameter counts are printed alongside a manual formula-based verification, and the fraction of total model parameters residing in convolutional layers is computed.

## Visualizations

The notebook produces:

- Sample images and class distribution bar chart
- Training/validation accuracy and loss curves
- Confusion matrix heatmap
- Grid of 25 sample predictions (correct/incorrect color-coded)
- Feature maps for each convolutional layer on a sample image
- Early-layer vs. deep-layer feature map comparison
- Max-pooling vs. average-pooling feature map comparison
- First-layer filter visualizations
- Bar charts comparing test accuracy across pooling type, kernel size, stride, padding, filter counts, optimizer, and batch size
- A summary table of all ablation results

## Requirements

```
numpy
pandas
matplotlib
tensorflow
scikit-learn
```

Install with:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn
```

A GPU is recommended — besides the 15-epoch baseline, the ablation study trains 15 additional model variants (5 epochs each).

## Usage

1. Open `Lab-3-Dl.ipynb` in Google Colab or Jupyter.
2. Run all cells top to bottom.
3. Review the baseline model's training curves, test metrics, and confusion matrix.
4. Explore the feature map and filter visualizations to see what the network has learned.
5. Review the ablation study's per-dimension bar charts and the final summary table.

## Project Structure

```
.
├── Lab-3-Dl.ipynb   # Main notebook: baseline CNN, interpretability, hyperparameter ablation study
└── README.md
```

## Author

Sanjay — B.Tech AI & Data Science, Shiv Nadar University Chennai
