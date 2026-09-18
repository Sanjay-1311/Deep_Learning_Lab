# CS3807 Deep Learning Lab 5 — MobileNetV2 Hyperparameter Study (Block-by-Block)

A block-by-block deep learning experiment (CS3807 Deep Learning Laboratory, Experiment 5) that builds a MobileNetV2-based classifier for the **Oxford-IIIT Pet** dataset (37 breeds) and systematically studies the effect of weight initialization, regularization, optimizers, learning rate, batch size, dropout, and feature extraction vs. fine-tuning — validated with stratified 5-fold cross-validation before a final, held-out test evaluation. This is an expanded, fully block-organized version of an earlier submitted lab (`lab_5_dl.py`).

## Overview

The notebook is split into 25 labeled blocks:

**Data pipeline (Blocks 1–6)**
1. Download the Oxford-IIIT Pet dataset via `kagglehub`
2. Parse breed labels from image filenames and inspect class distribution
3. Split into train (70%) / validation (15%) / test (15%) with stratification — **the test set is set aside and never touched during hyperparameter tuning**
4. Build a `tf.data` loading pipeline: read JPEG → decode → resize to 224×224 → MobileNetV2 preprocessing
5. Construct batched, shuffled, prefetched `train_ds` / `val_ds` / `test_ds`
6. Visually inspect sample batches

**Baseline & experiment infrastructure (Blocks 7–8)**
7. Train a baseline MobileNetV2 feature-extraction model (frozen base, `GlobalAveragePooling2D → Dense(128, ReLU) → Dense(37, Softmax)`) for 10 epochs
8. Define reusable helpers: a configurable `build_feature_extractor_model()` (initializer, regularization, dropout, batch norm), `compile_model()` (optimizer + learning rate), `plot_history()`, and `make_dataset()`

**Hyperparameter studies (Blocks 9–15)**

| Block | Study | Options compared |
|---|---|---|
| 9 | Weight initialization | zeros, random normal, Glorot, He |
| 10 | Regularization | none, L2, dropout, batch normalization |
| 11 | Batch normalization | with vs. without |
| 12 | Optimizer | SGD, SGD+momentum, RMSprop, Adam |
| 13 | Learning rate | 0.001, 0.0001 |
| 14 | Batch size | 16, 32, 64 |
| 15 | Dropout rate | 0.0, 0.25, 0.5 |

Each study trains fresh models for 10 epochs, plots the relevant loss/accuracy curves, and selects a "best" setting used going forward (`best_initializer`, `best_optimizer`, `best_lr`, `best_batch_size`, `best_dropout`).

**Feature extraction vs. fine-tuning (Block 16)**

Trains a feature-extraction model (frozen MobileNetV2 base) and a fine-tuned model (last 20 base layers unfrozen, BatchNorm layers kept frozen, learning rate 1e-5) and compares validation accuracy/loss curves.

**Cross-validation & final model (Blocks 17–24)**
17. Define four candidate configurations (C1–C4) combining the best settings found above, including one fine-tuning configuration
18. Run **stratified 5-fold cross-validation** (train+validation data only, test set excluded) for each configuration, reporting per-fold accuracy, mean, and standard deviation
19. Plot 5-fold CV accuracy with error bars and select the best configuration by mean accuracy
20. Retrain the best configuration on the **full** train+validation pool
21. Evaluate on the held-out test set for the first and only time: accuracy, weighted precision/recall/F1, parameter count, training time
22. Plot the confusion matrix (37×37 breed classes)
23. Display a sample of misclassified test images
24. Compile an overall results table summarizing CV accuracy, standard deviation, test accuracy, and training time

**Discussion questions (Block 25)**

Written answers to 23 conceptual questions covering parameters vs. hyperparameters, weight initialization, overfitting, dropout, batch normalization, optimizers, learning rate effects, batch size, convolution stride/padding, MobileNetV2's efficiency (depthwise separable convolutions, inverted residuals), transfer learning, and the role of cross-validation and test-set integrity.

## Dataset

- **Dataset:** [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) (downloaded via `kagglehub`, dataset `tanlikesmath/the-oxfordiiit-pet-dataset`)
- **Classes:** 37 cat and dog breeds, parsed from image filenames (e.g. `Abyssinian_1.jpg` → `Abyssinian`)
- **Split:** 70% train / 15% validation / 15% test, stratified by breed; the test split is held out from all tuning and only used once for final evaluation
- **Preprocessing:** images decoded from JPEG, resized to 224×224, and scaled via `mobilenet_v2.preprocess_input`

## Model Architecture

`Input(224, 224, 3) → MobileNetV2 (ImageNet weights, frozen or partially unfrozen) → GlobalAveragePooling2D → [optional BatchNorm] → Dense(128, activation, configurable init/L2) → [optional Dropout] → Dense(37, Softmax)`

Configurable dimensions: weight initializer, L2 regularization, dropout rate, batch normalization, optimizer, learning rate, batch size, and (for fine-tuning) how many of MobileNetV2's final layers are unfrozen (last 20, with BatchNorm layers kept frozen).

## Methodology Notes

- **Test-set integrity:** the independent test split is explicitly excluded from every hyperparameter search and cross-validation step, and is evaluated exactly once — after the winning configuration is selected and retrained on the full train+validation pool.
- **Cross-validation:** stratified 5-fold CV (`CV_EPOCHS = 3` for speed; the notebook notes this can be raised to 5–10 for a full run) is used to select among four candidate configurations before committing to a final model.
- Session state is cleared (`tf.keras.backend.clear_session()` + `gc.collect()`) between experiments to avoid memory buildup across the many models trained.

## Evaluation

Final reported metrics (from the retrained best configuration, on the held-out test set):

- Mean CV accuracy and standard deviation (from the winning configuration's 5 folds)
- Test accuracy, weighted precision, recall, F1-score
- Training time and total parameter count
- Confusion matrix over all 37 breeds
- Sample misclassified images

Exact values depend on the training run (weight initialization draws, data shuffling, hardware) and are produced by executing the notebook rather than fixed here.

## Visualizations

The notebook produces 14–15 plots covering:

- Training loss and validation accuracy by weight initialization
- Training/validation accuracy and loss by regularization scheme
- With vs. without batch normalization
- Training loss and validation accuracy by optimizer (plus a summary table)
- Learning rate, batch size, and dropout rate vs. validation accuracy
- Feature extraction vs. fine-tuning accuracy/loss curves
- 5-fold cross-validation accuracy with error bars
- Final confusion matrix and misclassified test images

## Requirements

```
tensorflow
tensorflow-datasets  # not required if using kagglehub path only
kagglehub
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install tensorflow kagglehub pandas numpy matplotlib seaborn scikit-learn
```

A Kaggle account/API access may be required for `kagglehub` to download the dataset. A GPU is strongly recommended — the notebook trains dozens of MobileNetV2-based models across all the sweeps, the 5-fold CV stage, and the final retraining.

## Usage

1. Open the notebook in Google Colab or Jupyter with a GPU runtime.
2. Run Blocks 1–6 to download the dataset, parse labels, and build the data pipeline.
3. Run Block 7 for the baseline model, then Block 8 to define the shared helper functions.
4. Run Blocks 9–16 in order — each hyperparameter study builds on the "best" values selected by the previous ones.
5. Run Blocks 17–19 for cross-validation and configuration selection.
6. Run Blocks 20–24 to retrain the best configuration and produce the final test-set evaluation, confusion matrix, and results table.
7. Block 25 contains written discussion answers — no code to run.

## Project Structure

```
.
├── lab_5_dl_block_by_block.ipynb   # Main notebook: data pipeline, hyperparameter studies, CV, final evaluation, discussion
└── README.md
```

## Author

Sanjay — B.Tech AI & Data Science, Shiv Nadar University Chennai
