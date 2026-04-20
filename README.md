# 📰 ShottiCheck: Machine Learning Based Fake News Detection System

ShottiCheck is a machine learning-based fake news detection system that uses Natural Language Processing (NLP) techniques to classify news articles as **real** or **fake**.
The system is built using TF-IDF vectorization and Logistic Regression, and includes an interactive web interface powered by Gradio.

---

## 🚀 Features

* 🧠 NLP-based text preprocessing and feature extraction
* 📊 TF-IDF with unigrams and bigrams
* ⚙️ Logistic Regression classifier
* 🌐 Interactive web UI using Gradio
* 💻 Command-line interface (CLI) fallback
* 📁 Model persistence using pickle

---

## 🧪 Tech Stack

* Python
* pandas, numpy
* scikit-learn
* Gradio
* pickle

---

## 📂 Dataset

This project uses the **WELFake Dataset** for training.

👉 Download from Kaggle:
https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

Place the file `WELFake_Dataset.csv` in the project root directory.

If the dataset is not found, the system automatically uses a built-in demo dataset.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/shotticheck-fake-news-detection.git
cd shotticheck-fake-news-detection
```

Install dependencies:

```bash
pip install pandas scikit-learn gradio requests
```

---

## ▶️ Usage

Run the project:

```bash
python fake_news_detector.py
```

### 🌐 Web Interface

* Launches a Gradio UI
* Paste news text and get prediction

### 💻 CLI Mode

If Gradio is not installed, the system falls back to command-line mode.

---

## 📊 Model Details

* **Vectorization:** TF-IDF (max_features=50,000, n-grams: 1–2)
* **Classifier:** Logistic Regression
* **Train/Test Split:** 80/20
* **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score

---

## 📸 Example

**Input:**

```
NASA confirms alien spacecraft found on dark side of the moon
```

**Output:**

```
🔴 FAKE NEWS — 98.2% confidence
Fake: 98.2% | Real: 1.8%
```

---

## ⚠️ Disclaimer

This project is intended for educational and research purposes only.
It does not guarantee 100% accuracy and should not be used as a sole source for verifying news authenticity.

---

## 📌 Future Improvements

* 🔄 Add deep learning models (LSTM, BERT)
* 🌍 Support Bangla language news
* 📱 Deploy as a web/mobile application
* 📈 Improve dataset diversity

---

## 👨‍💻 Author

**Ijbat Tahjib Bhuiyan**

---

## ⭐ Acknowledgements

* WELFake Dataset (Kaggle)
* scikit-learn documentation
* Gradio for UI framework

---
