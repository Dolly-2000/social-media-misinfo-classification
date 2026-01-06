# Social Media Misinformation Classification using Classical ML and BERT

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E)

## 📌 Project Overview
This project implements an end-to-end Natural Language Processing (NLP) pipeline to detect misinformation on social media platforms. The goal was to fairly benchmark **Classical Machine Learning** approaches against **State-of-the-Art Transformer (BERT)** models.

Using the **Constraint@AAAI-2021** dataset, I classified social media posts as either **Real** or **Fake**. The project involves extensive data preprocessing, hyperparameter tuning, and a comparative analysis of seven different algorithms.

## 📂 Dataset
*   **Source:** Constraint@AAAI-2021 Shared Task.
*   **Size:** ~10,700 social media posts (Twitter, Facebook, etc.).
*   **Classes:**
    *   `Real`: Verified information sources.
    *   `Fake`: Misinformation/Disinformation.
*   **Splits:** 64% Train, 16% Validation, 20% Test.

## 🛠️ Methodology

### 1. Data Preprocessing
*   **Cleaning:** Lowercasing, URL removal, hashtag handling, and whitespace normalization.
*   **Emoji Handling:** Converted emojis to text (e.g., 😂 -> `:joy:`) using the `emoji` library to retain semantic meaning.
*   **Vectorization:**
    *   **Classical ML:** TF-IDF Vectorization (Top 5000 features).
    *   **Transformers:** BERT-specific Tokenizers (WordPiece/SentencePiece).

### 2. Models Implemented
**Classical Machine Learning:**
1.  **K-Nearest Neighbors (KNN)** - Baseline distance-based classifier.
2.  **Logistic Regression** - Linear classifier.
3.  **Support Vector Machine (SVM)** - Linear Kernel.
4.  **K-Means Clustering** - Unsupervised analysis.
5.  **Neural Networks (MLP)** - Multi-layer Perceptron.
6.  **Ensemble Learning** - Gradient Boosting Classifier.

**Deep Learning (Transformers):**
Fine-tuned four specific BERT-based models for Sequence Classification:
1.  `bert-base-uncased`
2.  `digitalepidemiologylab/covid-twitter-bert` (Domain specific)
3.  `sarkerlab/SocBERT-base` (Social Media specific)
4.  `Twitter/twhin-bert-base` (Multilingual/Twitter specific)

## 📊 Results & Comparative Analysis

| Model Category | Algorithm | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- |
| **Classical ML** | KNN | ~90.0% | 0.90 |
| | **Logistic Regression** | **92.6%** | **0.93** |
| | SVM (Linear) | 92.2% | 0.92 |
| | Neural Network | 90.5% | 0.91 |
| | Gradient Boosting | 90.2% | 0.90 |
| **Transformers** | BERT Base | 94.7% | 0.95 |
| | Covid-Twitter-BERT | 97.2% | 0.97 |
| | SocBERT | 95.6% | 0.96 |
| | **Twhin-BERT** | **97.4%** | **0.975** |

**Key Findings:**
*   **Logistic Regression** outperformed more complex classical models like Neural Networks due to the high-dimensional sparse nature of TF-IDF vectors.
*   **Twhin-BERT** achieved the highest performance (97.4% Accuracy), demonstrating that pre-training on social media data (Twitter) significantly improves misinformation detection compared to generic BERT models.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Dolly-2000/social-media-misinfo-classification.git
    cd social-media-misinfo-classification
    ```

2.  **Install dependencies:**

3.  **Run the Notebooks:**
    You can run the notebooks in order. For the Transformer models, a GPU environment (like Google Colab T4) is recommended.

## 👨‍💻 Tech Stack
*   **Languages:** Python 3.11
*   **Libraries:** PyTorch, Scikit-learn, Transformers (Hugging Face), Pandas, NumPy, NLTK.
*   **Environment:** Jupyter Notebook / Google Colab.

## 📜 License
This project is for educational purposes. Dataset belongs to the authors of the Constraint@AAAI-2021 Shared Task.
