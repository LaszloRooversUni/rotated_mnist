# Rotated MNIST: Testing Rotation Invariance in CNNs

IMPORTANT: For the blogpost, see `blog.md`. This README serves only as a quick overview.

This project explores how rotation affects the performance of a simple CNN trained on MNIST.

Inspiration was taken from the CNN examples in the professor's lecture slides, and from studies such as:

- *Cohen & Welling, 2016* — [Group Equivariant Convolutional Networks](https://arxiv.org/abs/1602.07576)
- *Esteves et al., 2018* — [Polar Transformer Networks](https://arxiv.org/abs/1709.01889)

These papers propose architectural changes to improve rotation equivariance. In contrast, we analyze the behavior of a standard CNN when trained on unrotated data and tested on rotated variants, showing its limitations in a simple, intuitive setup.


## 🔍 Goal

To test whether a CNN trained on regular MNIST can handle rotated digits.

## 📁 Dataset

We use the standard MNIST dataset, then generate controlled variants with digits rotated by 30°, 60°, and 90°.

Run:

```bash
cd dataset_generation
python rotate_mnist.py
```

## 🧠 Model

A small CNN with two convolutional layers.

## 🚀 Usage

1. Train the model:

```bash
cd model
python train.py
```

2. Evaluate on rotated test sets:

```bash
python evaluate.py
```

## 📊 Results

| Angle (degrees) | Test Accuracy (%)  |
|---|---|
| 0 (original) | 98.94  |
| 30 | 85.29  |
| 60 | 30.91 |
| 90 | 13.71 |


## ⚠️ Disclaimer on Generative AI Usage

Parts of the code were auto-completed using GitHub Copilot. The documentation layout and structure (i.e. generating the markdown table) were assisted by OpenAI's ChatGPT.

---

Author: Lászlo Roovers
