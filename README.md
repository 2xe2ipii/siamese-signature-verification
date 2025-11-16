Siamese CNN for Offline Signature Verification

This project implements a Siamese Convolutional Neural Network for writer-independent offline signature verification, using TensorFlow and contrastive learning.

The system trains on pairs of images:

Positive pairs → two genuine signatures from the same writer

Negative pairs → one genuine + one forged signature

Then the network learns an embedding space where genuine pairs have low Euclidean distance and forged pairs have high distance.

📂 Project Structure
project/
│── main.py & main.ipynb
│── requirements.txt
│── README.md
│── .gitignore
│── models/          # auto-generated (ignored)
│── dataset/         # your signature dataset (ignored)

🧠 How the Dataset Must Be Structured

Your dataset folder should follow this exact directory structure:

dataset/
│── 001/
│── 001_forg/
│── 002/
│── 002_forg/
│── 003/
│── 003_forg/
...


Where:

001/ contains genuine signatures

001_forg/ contains forged signatures for writer 001

The code automatically pairs them.

🚀 How to Run Training

Make sure you are inside your conda TensorFlow GPU environment, then run:

python main.py


The script automatically:

Loads dataset

Splits writers into train/test sets

Generates genuine–genuine and genuine–forged pairs

Builds a Siamese CNN with L2-normalized embeddings

Trains using contrastive loss

Computes:

ROC curve + AUC

Balanced accuracy

Optimal distance threshold

Confusion matrix

Saves:

models/siamese_model.h5

models/embedding_model.h5

Plot images (loss curves, ROC, histograms)

metrics.json

📊 Outputs

Inside models/siamese_experiment_xx/ you will find:

siamese_model.h5

embedding_model.h5

metrics.json

learning_curve.png

roc_curve.png

distance_hist.png

confusion_matrix.png

🧪 Metrics Saved

The script writes a JSON file like:

{
  "best_threshold": 0.2981,
  "best_balanced_accuracy": 0.612,
  "roc_auc": 0.630,
  "confusion_matrix": [
    [320, 180],
    [210, 290]
  ],
  "num_train_pairs": 1000,
  "num_test_pairs": 1000
}

🛠 Features

Writer-independent training

Full experiment isolation (models/siamese_experiment_01, 02, …)

GPU optimized:

Mixed precision training

Euclidean distance in float32

Automatic loss scaling

L2-normalized embeddings for stability

Early Stopping + ReduceLROnPlateau

📌 Requirements

Install dependencies:

pip install -r requirements.txt


GPU support requires:

TensorFlow 2.13.0

CUDA 11.8

cuDNN 8.6+

🤝 Contributing

Feel free to fork this repository and experiment with:

Hard negative mining

Larger backbones (ResNet-50, MobileNet)

Margin tuning

Data augmentation

Different embedding sizes

📜 License

MIT License