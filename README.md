# 🎬 IMDB Sentiment Analysis – Binary Classification

> Comparing Bi-RNN, LSTM, Bi-LSTM, and BERT on IMDB movie reviews  
> 🧪 Focused on evaluating model performance for sentiment classification

---

## 📌 Project Description

This project investigates multiple deep learning architectures for **binary sentiment classification** on the **IMDB movie review dataset**. We compare the performance of:

- 🔁 Bi-directional RNN (Bi-RNN)
- 🌀 LSTM
- 🔁 Bi-directional LSTM (Bi-LSTM)
- 🤖 Pretrained BERT (uncased)

The goal is to determine how different models handle natural language understanding in the context of movie review sentiment.

---

## 🎯 Objective

To evaluate and compare deep learning models on their ability to classify IMDB movie reviews as either **positive** or **negative**, using both RNN-based and Transformer-based approaches.

---

## 📂 Dataset Information

- **Name**: *IMDB Dataset of 50K Movie Reviews*  
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
- **Included**: A local copy of `IMDB Dataset.csv` is included for convenience.  
- **Task**: Binary classification of review sentiment (positive/negative)  
- **Size**: 50,000 labeled movie reviews

---

## 🛠️ Tech Stack

### 🐍 Python Libraries

- **Data Handling**: `pandas`, `numpy`
- **Text Preprocessing**: `nltk`, `re`, `sklearn`
- **Deep Learning**: `torch`, `torch.nn`, `torchtext`, `transformers`
- **Model Training Utilities**: `tqdm`, `time`, `LabelEncoder`

### 🤖 BERT Model

- `bert-base-uncased` from HuggingFace Transformers

---

## 🚀 How to Run

> 💡 This project is implemented in **Jupyter Notebook**

### 🧰 Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git

Install dependencies:


pip install -r requirements.txt
Launch Jupyter Notebook and open the .ipynb file:


jupyter notebook
📊 Results
📌 All models were trained and evaluated using the same train-test split

🧪 BERT outperformed RNN-based models in accuracy and F1-score

📈 Metrics such as accuracy, precision, recall, and loss curves were used for comparison

🙏 Acknowledgements
Special thanks to the original dataset provider on Kaggle:
IMDB Dataset of 50K Movie Reviews

📄 License
This project is licensed under the terms defined in the LICENSE file.