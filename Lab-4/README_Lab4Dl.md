# CIFAR-10: Transfer Learning, Classic CNN Architectures & Hyperparameter Study

A three-part deep learning lab on CIFAR-10: (1) VGG16 transfer learning with a feature-extraction → fine-tuning workflow, (2) a from-scratch comparison of four classic CNN architectures (LeNet-5, AlexNet, GoogLeNet/Inception, ResNet50), and (3) a hyperparameter sweep over the VGG16 transfer-learning setup.

## Overview

The notebook (`Lab-4-Dl.ipynb`) is organized into three sections:

### 1. VGG16 Transfer Learning + Fine-Tuning
1. Load CIFAR-10 from a local `.npz` file and normalize pixels to `[0, 1]`
2. Visualize sample images and one-hot encode labels
3. Load **VGG16** (ImageNet weights, no top) with a frozen convolutional base
4. Add a new classifier head: `GlobalAveragePooling2D → Dense(128, ReLU) → Dense(10, Softmax)`
5. Train the head for 10 epochs (Adam, lr=0.001) with the base frozen; plot accuracy/loss curves; evaluate on the test set
6. **Fine-tune:** unfreeze the last ~4 layers of VGG16, recompile with a much lower learning rate (1e-5), and continue training for 10 more epochs
7. Compare pre- vs. post-fine-tuning test accuracy and plot the combined training history (with a marker for where fine-tuning starts)
8. Full evaluation: macro precision/recall/F1, classification report, confusion matrix heatmap, and a sample of misclassified images

### 2. Classic CNN Architecture Comparison
Four architectures are implemented from scratch (adapted for CIFAR-10's 32×32 inputs) and each trained for 10 epochs, tracking parameter count, test accuracy, and training time:

| Architecture | Key structure |
|---|---|
| **LeNet-5** | Conv(6,5x5) → AvgPool → Conv(16,5x5) → AvgPool → Dense(120) → Dense(84) → Dense(10) |
| **AlexNet** (adapted) | 5 conv layers (96→256→384→384→256 filters) with max pooling, → Dense(1024) ×2 with Dropout(0.5) → Dense(10) |
| **GoogLeNet** (adapted) | Stem conv + 3 Inception modules (parallel 1x1/3x3/5x5/pool-proj branches) → GlobalAveragePooling → Dropout(0.4) → Dense(10) |
| **ResNet50** | Pretrained ResNet50 (ImageNet weights, fully unfrozen/trainable) → GlobalAveragePooling → Dense(128) → Dense(10), with ResNet-specific preprocessing |

Results (parameter count, test accuracy, training time) for each architecture are collected into a shared `results` dictionary for comparison.

### 3. VGG16 Hyperparameter Study
A configurable `build_model()` / `run_experiment()` pair sweeps over the VGG16 transfer-learning setup, varying one factor at a time from a baseline configuration (lr=0.001, batch=32, epochs=10, Adam, 128 dense units, fully frozen base):

| Experiment | Change from baseline |
|---|---|
| Baseline | — |
| LR=0.0001 | Lower learning rate |
| Batch=16 / Batch=64 | Smaller / larger batch size |
| Epochs=20 | Longer training |
| Optimizer=SGD | SGD instead of Adam |
| Dense=256 | Wider classifier head |
| Frozen=Partial | Last ~4 VGG16 layers unfrozen instead of fully frozen |

Each run prints its configuration alongside test accuracy and training time.

## Dataset

- **Dataset:** CIFAR-10, loaded from a local `cifar10.npz` file
- **Classes (10):** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- **Preprocessing:** pixel values scaled to `[0, 1]`; labels one-hot encoded (10 classes); ResNet50 branch uses `resnet50.preprocess_input` on top of the rescaled images

## Evaluation

For the main VGG16 transfer-learning model:

- Training/validation accuracy and loss curves (separately for the frozen-base phase and combined with the fine-tuning phase)
- Test loss and accuracy, before and after fine-tuning
- Macro-averaged precision, recall, F1-score
- Full classification report and confusion matrix heatmap
- Misclassified image samples

For the architecture comparison and hyperparameter sweep, the key metrics are test accuracy, parameter count, and training time per configuration.

Since exact metric values depend on the actual training run (hardware, random initialization, etc.), they are produced by executing the notebook rather than fixed here.

## Visualizations

The notebook generates and saves:

- `accuracy_loss_plots.png` — training vs. validation accuracy/loss for the frozen-base phase
- `full_training_plots.png` — combined accuracy/loss across frozen-base + fine-tuning, with a marker at the fine-tuning transition
- `confusion_matrix.png` — confusion matrix heatmap on the test set
- `misclassified_images.png` — a sample of 10 misclassified test images with true/predicted labels
- Sample CIFAR-10 images with class labels (displayed inline)

## Requirements

```
tensorflow
numpy
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

> **Note:** The notebook expects a local `cifar10.npz` file with `x_train`, `y_train`, `x_test`, `y_test` arrays in the working directory. If you don't have one, it can be created from `tensorflow.keras.datasets.cifar10.load_data()` and saved with `np.savez`.

A GPU is strongly recommended: the notebook trains the VGG16 head, fine-tunes VGG16, and trains four additional architectures from scratch (including a fully-trainable ResNet50), plus 8 more VGG16 hyperparameter runs — a substantial amount of total compute.

## Usage

1. Place `cifar10.npz` in the working directory (see note above).
2. Open `Lab-4-Dl.ipynb` in Google Colab or Jupyter with a GPU runtime enabled.
3. Run the VGG16 transfer-learning + fine-tuning section first.
4. Run the classic-architecture comparison section to train LeNet-5, AlexNet, GoogLeNet, and ResNet50 and populate the `results` dictionary.
5. Run the hyperparameter sweep section to compare VGG16 configurations, populating `results_hp`.
6. Review the printed metrics, saved plots, and the `results` / `results_hp` collections for the final comparisons.

## Project Structure

```
.
├── Lab-4-Dl.ipynb              # Main notebook: VGG16 transfer learning, architecture comparison, hyperparameter sweep
├── cifar10.npz                 # Dataset (not included — see Usage)
├── accuracy_loss_plots.png     # Generated on run
├── full_training_plots.png     # Generated on run
├── confusion_matrix.png        # Generated on run
├── misclassified_images.png    # Generated on run
└── README.md
```

## Author

Sanjay — B.Tech AI & Data Science, Shiv Nadar University Chennai
