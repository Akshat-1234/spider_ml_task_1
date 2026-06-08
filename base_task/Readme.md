# Fashion-MNIST Classification

## Overview

This project implements the neural network architecture provided in the SPML Task 1 Base Task using PyTorch.

The model is trained on the Fashion-MNIST dataset and follows the architecture specified in the assignment, including the dual-branch structure and skip connection.

---

## Dataset

Fashion-MNIST contains 28×28 grayscale images belonging to 10 clothing categories.

Dataset split:

* Training: 50,000 samples
* Validation: 10,000 samples
* Test: 10,000 samples

---

## Model Architecture

```text
Input (28×28)
      │
Flatten (784)
      │
Linear(784 → 16)
      │
     Split
    /     \
16→8→8   16→12→8
  │
Skip Add
    \     /
  Concatenate
       │
Linear(16 → 10)
       │
    Output
```

---

## Training

* Optimizer: Adam
* Learning Rate: 0.001
* Loss Function: CrossEntropyLoss
* Epochs: 100

Training and validation metrics were tracked during training and visualized using Matplotlib.

---

## Results

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 85.83% |
| Precision | 85.84% |
| Recall    | 85.83% |
| F1 Score  | 85.82% |

---

## Generated Files

* `model_weights.pkl` – trained model weights
* `submission.csv` – predictions for the test set

---

## Project Structure

```text
base_task/
├── SPML_BASE_TASK.ipynb
├── model_weights.pkl
├── submission.csv
└── requirements.txt
```

---

## Tech Stack Used

* PyTorch
* Torchvision
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
