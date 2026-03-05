# LSTM Sentiment Analysis using PyTorch

## Overview

This project implements a **Long Short-Term Memory (LSTM) neural network** using PyTorch to perform **sentiment analysis on movie reviews** from the IMDB dataset.

The goal is to classify reviews as **positive or negative** by modeling sequential text data. The project demonstrates the complete deep learning pipeline including data preprocessing, vocabulary construction, sequence encoding, model training, evaluation, and visualization.

The final model achieves **over 80% accuracy** on the test dataset.

---

# Dataset

The dataset used in this project is the **IMDB Movie Review Dataset**, which contains labeled movie reviews for binary sentiment classification.

Dataset Source:
https://ai.stanford.edu/~amaas/data/sentiment/

Dataset structure:

```
aclImdb/
 ├── train/
 │   ├── pos/
 │   ├── neg/
 ├── test/
 │   ├── pos/
 │   ├── neg/
```

Each review is stored as a `.txt` file and labeled as:

* **Positive (1)**
* **Negative (0)**

Dataset size:

| Split    | Reviews |
| -------- | ------- |
| Training | 25,000  |
| Test     | 25,000  |

---

# Project Pipeline

The project follows a standard NLP deep learning workflow:

```
Raw Text Reviews
        ↓
Text Preprocessing
        ↓
Tokenization
        ↓
Vocabulary Construction
        ↓
Word → Index Encoding
        ↓
Sequence Padding
        ↓
PyTorch Dataset & DataLoader
        ↓
Embedding Layer
        ↓
LSTM Network
        ↓
Sentiment Prediction
```

---

# Text Preprocessing

Text reviews are cleaned and normalized before training.

Preprocessing steps include:

* Lowercasing text
* Removing special characters and punctuation
* Tokenization using NLTK
* Stopword removal
* Building a vocabulary from training data
* Converting tokens into numerical sequences

Sequences are padded to a fixed length of **200 tokens** for batch processing.

---

# Model Architecture

The model is implemented using **PyTorch** and consists of the following layers:

```
Embedding Layer
      ↓
LSTM Layer
      ↓
Dropout
      ↓
Fully Connected Layer
      ↓
Binary Sentiment Output
```

Model configuration:

| Layer     | Details                          |
| --------- | -------------------------------- |
| Embedding | 128-dimensional embeddings       |
| LSTM      | Hidden size = 256                |
| Dropout   | 0.5                              |
| Output    | 1 neuron (binary classification) |

Loss Function:

```
BCEWithLogitsLoss
```

Optimizer:

```
Adam
```

---

# Training

The model is trained for **10 epochs** using mini-batch gradient descent.

Batch size:

```
32
```

During training the following metrics are tracked:

* Training Loss
* Validation Loss
* Training Accuracy
* Validation Accuracy

---

# Results

Final evaluation on the test dataset produced the following metrics:

| Metric    | Score |
| --------- | ----- |
| Accuracy  | ~0.82 |
| Precision | ~0.82 |
| Recall    | ~0.82 |
| F1 Score  | ~0.82 |

The model successfully exceeds the **80% accuracy target**.

---

# Visualizations

Training progress and learned embeddings were visualized.

### Training Curves

Loss and accuracy were plotted across epochs.

Saved as:

```
plots/training_curves.png
```

### Word Embedding Visualization

A **t-SNE visualization** was generated to observe semantic relationships between learned word embeddings.

Saved as:

```
plots/tsne_embeddings.png
```

---

# Error Analysis

Some reviews were misclassified by the model. Analysis of these examples revealed common patterns:

* Reviews containing **mixed sentiment language**
* **Very long reviews** where important sentiment cues appear after truncation
* **Narrative or analytical writing styles** with implicit sentiment

Future improvements could include:

* Increasing sequence length
* Using **Bidirectional LSTM**
* Applying **attention mechanisms**
* Using pretrained embeddings such as **GloVe or Word2Vec**

---

# Repository Structure

```
lstm-sentiment-analysis/
│
├── notebook.ipynb
├── README.md
├── requirements.txt
├── sentiment_lstm_model.pth
│
├── plots/
│   ├── training_curves.png
│   ├── tsne_embeddings.png
```

---

# Installation

Clone the repository:

```
git clone https://github.com/ashifa-1/lstm-sentiment-analysis.git
cd lstm-sentiment-analysis
```

Create a virtual environment:

```
python -m venv venv
```

Activate environment:

Windows:

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Running the Project

Start Jupyter Notebook:

```
jupyter notebook
```

Open:

```
notebook.ipynb
```

Run the notebook from top to bottom to reproduce results.

