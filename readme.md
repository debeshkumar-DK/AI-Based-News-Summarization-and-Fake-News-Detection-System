# 🕵️ News Summarization and Fake News Detection using AI and Machine Learning

> **A Unified Platform for Combating Misinformation & Information Overload**
> *Minor Project (7th Semester)*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Framework](https://img.shields.io/badge/Framework-Flask-green) ![ML](https://img.shields.io/badge/AI-PyTorch%20%7C%20TensorFlow-orange) ![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
The exponential growth of digital media has introduced two pressing challenges: information overload and the rapid spread of fake news. This project addresses these issues by integrating **Natural Language Processing (NLP)** and **Machine Learning** into a unified web platform.

The system performs two core functions:
1.  **Fake News Detection:** Classifies articles as Real or Fake using a novel **Hybrid Transformer Architecture** that combines BERT embeddings with manual linguistic feature extraction.
2.  **Extractive Summarization:** Condenses lengthy articles into concise summaries using **Sentence-BERT (SBERT)** and **K-Means Clustering**.

---

## 🧠 Model Architectures & Features

### 1. Fake News Detection (Hybrid Approach)
We implemented a **Hybrid Model** that does not rely solely on "Black-box" deep learning. It feeds two inputs into the classification layer: **BERT Contextual Embeddings** + **Linguistic Features**.

#### 🅰️ Hybrid-25 Model (Advanced - 99.86% Acc)
Uses **25 Features** covering structure, sentiment, and readability:
* **Structural (7):** Character Count, Word Count, Sentence Count, Avg Word Length, Avg Sentence Length, Vocabulary Richness, Digit Ratio.
* **Stylistic (5):** Uppercase Ratio, Punctuation Ratio, Special Char Ratio, Exclamation Ratio (`!`), Question Ratio (`?`).
* **Readability (6):** Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog, Smog Index, Automated Readability Index (ARI), Coleman-Liau Index.
* **Sentiment (6):** Polarity, Subjectivity (TextBlob), VADER Positive, Negative, Neutral, Compound scores.
* **Morphology (1):** Original text length.

#### 🅱️ Hybrid-8 Model (Lightweight - 99.9% Acc)
Uses **8 Key Features** focusing on emotion and complexity:
1.  **Flesch Reading Ease** (Readability)
2.  **Gunning Fog Index** (Complexity)
3.  **Sentiment Polarity** (Tone)
4.  **Subjectivity** (Opinion vs Fact)
5.  **Fear Score** (NRCLex)
6.  **Anger Score** (NRCLex)
7.  **Trust Score** (NRCLex)
8.  **Type-Token Ratio (TTR)** (Vocabulary Diversity)

### 2. Extractive Summarization (SBERT)
We implemented two SBERT-based models for summarization. The algorithm uses **K-Means Clustering** to select diverse sentences that represent different topics in the article.

* **✨ SBERT V2 (MPNet):** Fine-tuned on the CNN/DailyMail dataset using **Triplet Loss** for high semantic accuracy.
* **⚡ SBERT Base (MiniLM):** A lighter, faster model optimized for real-time CPU inference.

---

## 📊 Performance Evaluation

### Detection Models (Test Set Metrics)
*Evaluated on the ISOT and Indian Fake News datasets.*

| Model Architecture | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **✨ Hybrid BERT (25 Features)** | **99.86%** | **1.00** | **1.00** | **1.00** |
| Hybrid BERT (8 Features) | 99.90% | 1.00 | 0.99 | 1.00 |
| Frozen BERT + Dense NN | 85.87% | 0.82 | 0.91 | 0.87 |
| Frozen BERT + Sklearn | 95.16% | 0.96 | 0.95 | 0.95 |

*Note: The Dense NN model (85.87%) exhibits higher recall for fake news (0.91) but lower precision compared to the Hybrid models, demonstrating the value of adding linguistic features.*

### Summarization Models (ROUGE Scores)
*Evaluated on CNN/DailyMail Test Split.*

| Model | ROUGE-1 (Unigram) | ROUGE-2 (Bigram) | ROUGE-L (Sequence) | Inference Time |
| :--- | :---: | :---: | :---: | :---: |
| **✨ SBERT V2 (MPNet + K-Means)** | **0.3447** | **0.1228** | **0.2151** | ~1.5s |
| SBERT Base (MiniLM) | 0.3364 | 0.1145 | 0.2098 | **~0.4s** |

*Analysis: SBERT V2 consistently outperforms the Base model in capturing semantic meaning (Higher ROUGE-2/L), justifying the fine-tuning process.*

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Web Framework:** Flask
* **Deep Learning:** PyTorch, TensorFlow (Dense Model), Hugging Face Transformers
* **Machine Learning:** Scikit-Learn (K-Means, Logistic Regression)
* **NLP Libraries:** NLTK, TextBlob, TextStat, VaderSentiment, NRCLex
* **Deployment:** Ngrok (Tunneling)
* **Frontend:** HTML5, Tailwind CSS

---

## ⚙️ How to Run This Project

Follow these simple steps to set up and run the app.py on your local machine.

### Step 1: Create the Virtual Environment

Run this command to create an isolated space for your libraries (so they don't mess up your computer).

python -m venv venv

### Step 2: Activate the Virtual Environment

You must turn it on before installing anything.


For Windows:
venv\Scripts\activate


For Mac / Linux:
source venv/bin/activate

### Step 4: Set ngrok Authtoken 
Run this code inside the venv in the terminal

ngrok config add-authtoken YOUR_TOKEN_HERE

you can get your ngrok Authtoken here : https://dashboard.ngrok.com/get-started/your-authtoken

### Step 3: Install Dependencies

Now, use the requirements.txt file you created to install all the tools at once.

pip install -r requirements.txt

### Step 4: Run the App

Finally, start your website.

python app.py

### Step 5: Open in Browser
You will see a message like Running on http://127.0.0.1:5000. Hold Ctrl and click that link (or copy-paste it into Chrome or Any other Web Browser).

