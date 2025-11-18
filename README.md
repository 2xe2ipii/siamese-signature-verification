# Signature Verification Using Siamese Convolutional Neural Network (CNN)

**Data Science 2 Final Project** *Focus: Hyperparameter Optimization and Metric Learning Analysis*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

## 📌 Project Overview
This project implements a **Siamese Convolutional Neural Network (CNN)** for offline handwritten signature verification. Unlike traditional classification models, this architecture utilizes a **One-Shot Learning** approach to determine the authenticity of a signature by comparing it against a reference sample.

The core objective of this repository is **hyperparameter experimentation**. The code is designed to rigorously test how different configurations (Margins, Batch Sizes, Learning Rates) affect the stability of the Contrastive Loss function and the separation of genuine vs. forged embeddings.

## 👥 Authors
* **Abanilla, Tres Ynman**
* **De Silva, Arturo Andres**
* **Navarro, Radian**
* **Reyes, Drexler**
* **Viñas, Louis Evan Gabriel**
* **Yao, Josh Gareth**

---

## 📂 Dataset Structure
The system expects a dataset structured by identity. The code parses folder names to distinguish between genuine and forged signatures automatically.

**Required Directory Layout:**
```text
dataset/
├── 001/              # Genuine signatures for Person 001
│   ├── 1-001_01.jpg
│   └── ...
├── 001_forg/         # Forged signatures for Person 001
│   ├── 1-001_01_forg.jpg
│   └── ...
├── 002/
├── 002_forg/
└── ...
