# Single-Layer Perceptron — Banknote Authentication

A from-scratch, pure-Python/NumPy implementation of a single-layer Perceptron (no classes, no ML framework), trained with the classic online update rule to classify banknotes as authentic or forged using the UCI Banknote Authentication dataset.

## Overview

The notebook (`SIngle_layered_perceptron.ipynb`) walks through:

1. Load the banknote authentication dataset from a local CSV (`data_banknote_authentication.txt`)
2. Inspect the data: column names, missing values, summary statistics
3. Exploratory data analysis: feature histograms, correlation heatmap, pairwise scatter plots colored by class, per-feature boxplots by class
4. Standardize the four features with `StandardScaler`
5. Split into train/test sets (80/20)
6. Implement a perceptron manually — explicit weights/bias, a step activation function, and a hand-written training loop with the perceptron update rule
7. Train for 100 epochs, tracking misclassification count per epoch and keeping the best-performing weights/bias
8. Predict on the test set and compute a confusion matrix, accuracy, precision, recall, and F1-score by hand

## Dataset

- **Source:** UCI Banknote Authentication dataset (loaded from a local file: `data_banknote_authentication.txt`)
- **Features:** `variance`, `skewness`, `curtosis`, `entropy` (extracted from wavelet-transformed banknote images)
- **Target:** `authenticity` — 0 (authentic) or 1 (forged)
- **No missing values**

## Perceptron Implementation

The perceptron is implemented explicitly, without a class wrapper:

- **Weights:** initialized manually to `[1.0, 0.4, 0.2, 0.01]`, bias `0`
- **Activation:** a step function — outputs `1` if the weighted sum is `>= 0`, else `0`
- **Training loop:** for each sample, compute the pre-activation (`w · x + b`), apply the step function, compute the error (`y_true - y_pred`), and update each weight and the bias by `learning_rate * error * x`
- **Hyperparameters:** `epochs = 100`, `learning_rate = 0.001`
- **Model selection:** the epoch with the fewest misclassified training samples is kept as the best model (best weights, bias, and epoch number are tracked throughout training)

## Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.9855 |
| Precision | 0.9843 |
| Recall    | 0.9843 |
| F1-score  | 0.9843 |

**Confusion Matrix** (rows = actual, columns = predicted; `[TN, FP], [FN, TP]`):

```
[[146   2]
 [  2 125]]
```

The best training-epoch model was found at **epoch 45**, with **6 misclassified** training samples out of 1,097.

## Visualizations

The notebook produces:

- Histograms of all four features
- A correlation heatmap of the features
- Scatter plots of `variance` vs. `skewness`, `variance` vs. `curtosis`, `variance` vs. `entropy`, and `skewness` vs. `entropy`, each colored by class
- Boxplots of each feature grouped by class

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Place `data_banknote_authentication.txt` in the same directory the notebook expects (`/content/` if running on Google Colab, or update the path for local use).
2. Open `SIngle_layered_perceptron.ipynb` in Jupyter or Google Colab.
3. Run all cells top to bottom.
4. Review the printed per-epoch training log, the best-model summary, and the final confusion matrix and metrics.

## Project Structure

```
.
├── SIngle_layered_perceptron.ipynb   # Main notebook: EDA, from-scratch perceptron, evaluation
├── data_banknote_authentication.txt  # Dataset (not included — see Usage)
└── README.md
```

## Author

Sanjay — B.Tech AI & Data Science, Shiv Nadar University Chennai
