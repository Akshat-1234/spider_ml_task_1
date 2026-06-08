# Fashion-MNIST Autoencoder (Bonus Task)

This notebook implements an MLP-based Autoencoder on the Fashion-MNIST dataset.

The goal of the model is to learn a compressed representation of clothing images and reconstruct the original image from that compressed representation.

---

## Model Architecture

Encoder:

784 → 256 → 128 → Latent Dimension

Decoder:

Latent Dimension → 128 → 256 → 784

ReLU activations are used in the hidden layers and a Sigmoid activation is used in the final output layer.

---

## Training

- Dataset: Fashion-MNIST
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Epochs: 100
- Batch Size: 2048

The model is trained to minimize the reconstruction error between the original image and the reconstructed image.

---

## Experiments

Different bottleneck sizes were tested:

| Latent Dimension |
|------------------|
| 8 |
| 16 |
| 32 |
| 64 |
| 128 |

The objective was to study how the size of the latent representation affects reconstruction quality.

---

## Results

Validation loss generally decreased as the latent dimension increased.

A latent dimension of 8 resulted in the highest reconstruction error due to aggressive compression.

Larger latent dimensions preserved more information and produced better reconstructions, though the improvements became smaller beyond 32 dimensions.

---

## Visualizations

The notebook includes:

- Training and validation loss curves
- Original vs reconstructed image comparisons
- Latent dimension vs validation loss analysis

---

## Running the Notebook

Install dependencies:

```bash
pip install -r requirements.txt