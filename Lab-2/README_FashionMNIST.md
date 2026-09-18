# Fashion-MNIST Classification with a Tunable MLP

A fully-connected neural network (MLP) classifier for the **Fashion-MNIST** dataset, built with TensorFlow/Keras, evaluated as a baseline model and then improved through randomized hyperparameter search using SciKeras + scikit-learn's `RandomizedSearchCV`.

## Overview

The script (exported from a Colab notebook) covers:

1. Load Fashion-MNIST via `tensorflow.keras.datasets`
2. Visualize sample images per class and the class distribution across the training set
3. Preprocess: flatten 28×28 images to 784-dim vectors, normalize pixels to `[0, 1]`, one-hot encode labels
4. Train a **baseline MLP** (`Dense(128, relu) → Dense(64, relu) → Dense(10, softmax)`) with the Adam optimizer for 20 epochs
5. Plot baseline training/validation accuracy and loss curves
6. Evaluate the baseline model: accuracy, weighted precision/recall/F1, confusion matrix, per-class classification report
7. Define a configurable `create_model()` builder (variable hidden layers, neurons, learning rate, optimizer, activation, dropout)
8. Wrap the builder with `KerasClassifier` (SciKeras) and run a **randomized hyperparameter search** (`RandomizedSearchCV`, 20 iterations, 5-fold CV) over layer count, neuron count, learning rate, optimizer, activation function, dropout rate, batch size, and epoch count
9. Retrieve the best hyperparameters and evaluate the tuned model on the test set (accuracy, precision, recall, F1, confusion matrix, classification report)
10. Retrain a final model from scratch using the best hyperparameters and plot its training/validation accuracy and loss curves

## Dataset

- **Dataset:** Fashion-MNIST (`tensorflow.keras.datasets.fashion_mnist`)
- **Classes (10):** T-shirt/Top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot
- **Size:** 60,000 training images / 10,000 test images, 28×28 grayscale
- **Preprocessing:** flattened to 784-dim vectors, pixel values scaled to `[0, 1]`, labels one-hot encoded

## Model Architecture

**Baseline model:**

`Input(784) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)`

- Optimizer: Adam (default learning rate)
- Loss: categorical cross-entropy
- Epochs: 20, batch size: 32

**Tunable model (`create_model`):**

`Input(784) → [Dense(hidden_neurons, activation) → optional Dropout] × hidden_layers → Dense(10, Softmax)`

Hyperparameter search space:

| Hyperparameter | Values searched |
|---|---|
| Hidden layers | 1, 2, 3 |
| Hidden neurons | 32, 64, 128, 256 |
| Learning rate | 0.1, 0.01, 0.001 |
| Optimizer | Adam, SGD, RMSprop |
| Activation | ReLU, tanh, sigmoid |
| Dropout rate | 0.0, 0.2, 0.5 |
| Batch size | 16, 32, 64, 128 |
| Epochs | 10, 20, 30 |

Search strategy: `RandomizedSearchCV` with 20 sampled combinations and 5-fold cross-validation, scored on accuracy.

## Evaluation

Both the baseline and the tuned (best-hyperparameter) models are evaluated on the test set with:

- Accuracy, weighted precision, weighted recall, weighted F1-score
- Confusion matrix (baseline printed as an array; tuned model plotted as a heatmap)
- Full per-class classification report

The tuned model's best hyperparameters and cross-validation score are printed after the search completes, and the final model is retrained from scratch on those settings (with a 20% validation split) to produce its own accuracy/loss curves.

Since exact accuracy/precision/recall/F1 values and the winning hyperparameter combination depend on the randomized search's random draws and training run, they are produced by running the script rather than fixed here.

## Visualizations

The script generates and saves (as `.eps` files, 600 dpi) the following plots:

- Sample images from each Fashion-MNIST class
- Class distribution bar chart
- Baseline model: training accuracy, validation accuracy, training loss, and validation loss vs. epoch
- Confusion matrix heatmap for the tuned model
- Final (best-hyperparameter) model: training accuracy, validation accuracy, training loss, and validation loss vs. epoch

## Requirements

```
numpy
pandas
matplotlib
seaborn
tensorflow
scikit-learn
scikeras
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn scikeras
```

> **Note:** The script pins/re-installs specific versions during the session (`scikeras==0.13.0`, `scikit-learn==1.6.1`, later `scikit-learn==1.5.2`) to work around SciKeras/scikit-learn compatibility issues. If running outside Colab, install compatible versions of `scikeras` and `scikit-learn` up front rather than switching versions mid-run.

The hyperparameter search trains up to 20 × 5 = 100 model fits, so a GPU is recommended and the search may still take a significant amount of time on CPU.

## Usage

1. Run the script/notebook cells top to bottom in Google Colab (or a Jupyter environment with the packages above installed).
2. The baseline model trains and is evaluated first.
3. The randomized hyperparameter search runs next — this is the most time-consuming step.
4. Review `random_search.best_params_` and `random_search.best_score_`, then the final retrained model's evaluation and plots.

## Project Structure

```
.
├── fashion_mnist_mlp.py   # Main script: baseline MLP, hyperparameter search, final model, evaluation
└── README.md
```

## Author

Sanjay — B.Tech AI & Data Science, Shiv Nadar University Chennai
