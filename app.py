# ============================================================
# app.py — Fake News Detection & Summarization (v2.2 DUAL SBERT)
# ============================================================

import os
import re
import string
import joblib
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# ============================================================
# ENV + HUGGINGFACE AUTH
# ============================================================

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, login

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
# Fallback if .env fails
if not HF_TOKEN:
    HF_TOKEN = "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"

login(token=HF_TOKEN)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {DEVICE}")

# ============================================================
# NLTK + NLP LIBRARIES
# ============================================================

import nltk
# Force download all required NLTK data
for res in ["punkt", "wordnet", "stopwords", "punkt_tab", "averaged_perceptron_tagger"]:
    nltk.download(res, quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import textstat
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# ============================================================
# TRANSFORMERS / SBERT
# ============================================================

from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from sentence_transformers import SentenceTransformer

# ============================================================
# FLASK INIT
# ============================================================

app = Flask(__name__)

# ============================================================
# SHARED TOKENIZER
# ============================================================

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ============================================================
# MODEL 1 — HYBRID 8 FEATURES (PyTorch)
# ============================================================

class BertWithCustomFeatures(nn.Module):
    def __init__(self, num_features=8):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768 + num_features, 2)

    def forward(self, input_ids, attention_mask, features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        combined = torch.cat([pooled, features], dim=1)
        return self.classifier(combined), pooled

print("⏳ Loading Hybrid-8 model...")
try:
    hybrid8 = BertWithCustomFeatures().to(DEVICE)
    hybrid8.load_state_dict(
        torch.load(
            hf_hub_download(
                repo_id="DarkKnight001/hybrid-fake-news-detector-V2",
                filename="hybrid_fake_news_model.pth",
                token=HF_TOKEN
            ),
            map_location=DEVICE
        )
    )
    hybrid8.eval()
    
    scaler8 = joblib.load(
        hf_hub_download(
            repo_id="DarkKnight001/hybrid-fake-news-detector-V2",
            filename="scaler.pkl",
            token=HF_TOKEN
        )
    )
    print("✅ Hybrid-8 loaded")
except Exception as e:
    print(f"❌ Hybrid-8 Failed: {e}")
    hybrid8, scaler8 = None, None

# =========================
# CLASSES FOR HYBRID-25
# =========================

class PreProcessingFeatureExtractor:
    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()

    def extract_all_features(self, text):
        import string
        text = str(text)

        if not text:
            return {'original_length': 0, 'caps_ratio': 0, 'punctuation_ratio': 0, 'special_char_ratio': 0, 'digit_ratio': 0, 'exclamation_ratio': 0, 'question_ratio': 0}

        uppercase_count = sum(1 for c in text if c.isupper())
        punct_count = sum(1 for c in text if c in string.punctuation)
        special_count = sum(1 for c in text if c in "!@#$%^&*")
        digit_count = sum(1 for c in text if c.isdigit())

        return {
            "original_length": len(text),
            "caps_ratio": uppercase_count / len(text),
            "punctuation_ratio": punct_count / len(text),
            "special_char_ratio": special_count / len(text),
            "digit_ratio": digit_count / len(text),
            "exclamation_ratio": text.count("!") / len(text),
            "question_ratio": text.count("?") / len(text),
        }


class TextPreprocessor:
    def __init__(self):
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        import re
        text = str(text)
        text = re.sub(r"^[A-Z][A-Z\s,]+\([A-Za-z\s]+\)\s*[-–—]\s*", "", text)
        text = re.sub(r"^[A-Z][A-Z\s]+\s*[-–—]\s*", "", text)
        text = re.sub(r"^By\s+[A-Z][\w\s]+[-,]\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        return " ".join(text.split())

    def preprocess(self, text):
        from nltk.tokenize import word_tokenize
        text = self.clean_text(text).lower()
        tokens = word_tokenize(text)
        return " ".join(self.lemmatizer.lemmatize(t) for t in tokens)


class PostProcessingFeatureExtractor:
    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()

    def extract_all_features(self, text):
        from nltk.tokenize import sent_tokenize
        import numpy as np
        import textstat
        from textblob import TextBlob

        words = text.split()
        sentences = sent_tokenize(text)

        features = {
            "word_count": len(words),
            "char_count": len(text),
            "sentence_count": len(sentences),
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "vocabulary_richness": len(set(words)) / len(words) if words else 0,
        }

        # Safe TextStat Calls
        scorers = [
            (textstat.flesch_reading_ease, 'flesch_reading_ease'),
            (textstat.flesch_kincaid_grade, 'flesch_kincaid_grade'),
            (textstat.gunning_fog, 'gunning_fog'),
            (textstat.smog_index, 'smog_index'),
            (textstat.automated_readability_index, 'automated_readability_index'),
            (textstat.coleman_liau_index, 'coleman_liau_index')
        ]
        for func, name in scorers:
            try: features[name] = func(text)
            except: features[name] = 0

        blob = TextBlob(text)
        vader = self.vader.polarity_scores(text)

        features.update({
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity,
            "vader_positive": vader["pos"],
            "vader_negative": vader["neg"],
            "vader_neutral": vader["neu"],
            "vader_compound": vader["compound"],
        })
        return features

# ============================================================
# MODEL 2 — SKLEARN CLASSIFIER (ON BERT EMBEDDINGS)
# ============================================================

print("⏳ Loading Sklearn model...")
try:
    model_sklearn = joblib.load(
        hf_hub_download(
            repo_id="DarkKnight001/sklearn-detection-bert-model",
            filename="bert_sklearn.pkl",
            token=HF_TOKEN
        )
    )
    print("✅ Sklearn model loaded")
except Exception as e:
    print(f"❌ Sklearn Failed: {e}")
    model_sklearn = None

# ============================================================
# MODEL 3 — DENSE TENSORFLOW MODEL (OPTIONAL)
# ============================================================

model_dense = None
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text as text

    print("⏳ Loading Dense TF model...")
    
    # 1. Download the file path
    path_dense = hf_hub_download(
        repo_id="DarkKnight001/detection-bert-model",
        filename="bert.pkl",
        token=HF_TOKEN
    )
    
    # 2. FIX: Use joblib to load the .pkl file (NOT load_model)
    model_dense = joblib.load(path_dense)
    
    print("✅ Dense TF model loaded")
except Exception as e:
    print(f"⚠️ Dense model skipped: {e}")

# ============================================================
# MODEL 4 — HYBRID 25 FEATURES (ADVANCED)
# ============================================================

print("⏳ Loading Hybrid-25 model...")

try:
    HF_REPO_25 = "DarkKnight001/hybrid-bert-fake-news-claude"

    tokenizer25 = BertTokenizer.from_pretrained(HF_REPO_25)
    model_config = torch.load(
        hf_hub_download(repo_id=HF_REPO_25, filename="model_config.pt", token=HF_TOKEN)
    )

    NUM_FEATURES_25 = model_config["num_numerical_features"]
    MODEL_NAME_25 = model_config["bert_model"]

    class Hybrid25(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = BertForSequenceClassification.from_pretrained(MODEL_NAME_25, num_labels=2)
            self.feature_combiner = nn.Linear(self.bert.config.hidden_size + NUM_FEATURES_25, 256)
            self.layer_norm = nn.LayerNorm(256)
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Linear(256, 2)

        def forward(self, input_ids, attention_mask, numerical_features):
            outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
            combined = torch.cat([cls_output, numerical_features.float()], dim=1)
            x = torch.relu(self.feature_combiner(combined))
            x = self.layer_norm(x)
            x = self.dropout(x)
            return self.classifier(x)

    hybrid25 = Hybrid25().to(DEVICE)
    hybrid25.load_state_dict(
        torch.load(
            hf_hub_download(repo_id=HF_REPO_25, filename="pytorch_model.bin", token=HF_TOKEN),
            map_location=DEVICE
        ),
        strict=True
    )
    hybrid25.eval()

    scaler25 = joblib.load(hf_hub_download(repo_id=HF_REPO_25, filename="feature_scaler.pkl", token=HF_TOKEN))
    pre_extractor = joblib.load(hf_hub_download(repo_id=HF_REPO_25, filename="pre_extractor.pkl", token=HF_TOKEN))
    post_extractor = joblib.load(hf_hub_download(repo_id=HF_REPO_25, filename="post_extractor.pkl", token=HF_TOKEN))
    preprocessor = joblib.load(hf_hub_download(repo_id=HF_REPO_25, filename="preprocessor.pkl", token=HF_TOKEN))

    print("✅ Hybrid-25 loaded")
except Exception as e:
    print(f"❌ Hybrid-25 Failed: {e}")
    hybrid25, scaler25 = None, None

# ============================================================
# SBERT SUMMARIZATION: LOAD BOTH MODELS
# ============================================================

print("⏳ Loading Summarization Models...")
try:
    # Model 1: Your Fine-Tuned MPNet (V2) - High Accuracy
    sbert_v2 = SentenceTransformer("DarkKnight001/SBERT-Summarization-v2", token=HF_TOKEN)
    print("✅ SBERT V2 (MPNet) loaded")
    
    # Model 2: Fast Base Model (MiniLM) - High Speed
    sbert_base = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ SBERT Base (MiniLM) loaded")

except Exception as e:
    print(f"⚠️ Error loading SBERT models: {e}")
    # Fallback to base if V2 fails
    sbert_base = SentenceTransformer('all-MiniLM-L6-v2')
    sbert_v2 = sbert_base

def summarize(text, model_type="sbert_v2", num_sentences=3):
    """
    Summarizes text using the selected SBERT model + K-Means Clustering.
    """
    # 1. Select the model based on user input
    model = sbert_v2 if model_type == "sbert_v2" else sbert_base
    
    sentences = sent_tokenize(text)
    
    # 2. Safety Checks
    if len(sentences) == 0:
        return ""
    if len(sentences) <= num_sentences:
        return text
    
    # 3. Encode (Using selected model)
    embeddings = model.encode(sentences)
    
    # 4. K-Means Clustering
    num_clusters = min(num_sentences, len(sentences))
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(embeddings)
    
    # 5. Find Key Sentences
    closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
    closest_indices = sorted(closest_indices)
    
    # 6. Build Summary
    selected_sentences = [sentences[i] for i in closest_indices]
    return " ".join(selected_sentences)

# ============================================================
# FEATURE HELPERS
# ============================================================

def extract_8_features(text):
    blob = TextBlob(text)
    words = text.split()
    emotions = NRCLex(text).affect_frequencies
    return [
        textstat.flesch_reading_ease(text),
        textstat.gunning_fog(text),
        blob.sentiment.polarity,
        blob.sentiment.subjectivity,
        emotions.get("fear", 0),
        emotions.get("anger", 0),
        emotions.get("trust", 0),
        len(set(words)) / len(words) if words else 0
    ]

# ============================================================
# ROUTES
# ============================================================
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form.get("news", "").strip()
    model_choice = request.form.get("detection_model", "hybrid25")
    # ✅ GET THE SUMMARIZER CHOICE FROM FORM
    summarizer_choice = request.form.get("summarization_model", "sbert_v2") 
    
    # DEBUG: Print what we received
    print(f"\n🔍 DEBUG: Received request. Model: {model_choice}, Summarizer: {summarizer_choice}")

    if not text:
        return render_template("index.html", error="Text is empty")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(DEVICE)

    label, confidence = "Error", "0.00%"

    try:
        with torch.no_grad():
            # =========================
            # HYBRID 8 FEATURES
            # =========================
            if model_choice == "hybrid8" or model_choice == "hybrid_old":
                if hybrid8 and scaler8:
                    feats = scaler8.transform([extract_8_features(text)])
                    logits, _ = hybrid8(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        torch.tensor(feats, dtype=torch.float).to(DEVICE)
                    )
                    probs = torch.softmax(logits, dim=1)[0]
                    label = "Fake" if probs[1] > probs[0] else "Real"
                    confidence = f"{max(probs).item()*100:.2f}%"
                else: label = "Hybrid-8 Unavailable"

            # =========================
            # HYBRID 25 FEATURES
            # =========================
            elif model_choice == "hybrid25" or model_choice == "hybrid_new":
                if hybrid25 and scaler25:
                    pre = pre_extractor.extract_all_features(text)
                    clean = preprocessor.preprocess(text)
                    post = post_extractor.extract_all_features(clean)
                    all_f = {**pre, **post}
                    vec = [[all_f.get(f, 0) for f in scaler25.feature_names_in_]]
                    feats = scaler25.transform(vec)

                    enc = tokenizer25(clean, return_tensors="pt", truncation=True, padding=True, max_length=512)
                    logits = hybrid25(
                        enc["input_ids"].to(DEVICE),
                        enc["attention_mask"].to(DEVICE),
                        torch.tensor(feats, dtype=torch.float).to(DEVICE)
                    )
                    probs = torch.softmax(logits, dim=1)[0]
                    label = "Real" if probs[1] > probs[0] else "Fake"
                    confidence = f"{max(probs).item()*100:.2f}%"
                else: label = "Hybrid-25 Unavailable"

            # =========================
            # SKLEARN
            # =========================
            elif model_choice == "sklearn":
                if model_sklearn and hybrid8:
                    dummy = torch.zeros((1, 8)).to(DEVICE)
                    _, pooled = hybrid8(inputs["input_ids"], inputs["attention_mask"], dummy)
                    probs = model_sklearn.predict_proba(pooled.cpu().numpy())[0]
                    label = "Fake" if probs[1] > probs[0] else "Real"
                    confidence = f"{max(probs) * 100:.2f}%"
                else: label = "Sklearn Unavailable"

            # =========================
            # DENSE
            # =========================
            elif model_choice == "dense":
                if model_dense and hybrid8:
                    dummy = torch.zeros((1, 8)).to(DEVICE)
                    _, pooled = hybrid8(inputs["input_ids"], inputs["attention_mask"], dummy)
                    p = model_dense.predict(pooled.cpu().numpy())[0][0]
                    label = "Fake" if p > 0.5 else "Real"
                    confidence = f"{(p if p > 0.5 else 1-p) * 100:.2f}%"
                else: label = "Dense Unavailable"

    except Exception as e:
        import traceback
        traceback.print_exc()
        label = f"Err: {str(e)[:20]}"

    return render_template(
        "index.html",
        prediction=label,
        confidence=confidence,
        # ✅ PASS THE CHOSEN MODEL TO THE SUMMARIZE FUNCTION
        summary=summarize(text, model_type=summarizer_choice, num_sentences=3),
        original_text=text,
        selected_model=model_choice,
        selected_summarizer=summarizer_choice # ✅ Keep dropdown selected
    )

# ============================================================
# RUN WITH NGROK
# ============================================================

if __name__ == "__main__":
    from pyngrok import ngrok
    try:
        public_url = ngrok.connect(5000).public_url
        print(f"\n==================================================================")
        print(f" 🚀 PUBLIC URL: {public_url}")
        print(f"==================================================================\n")
    except Exception as e:
        print(f"⚠️ Ngrok Warning: {e}")

    app.run(port=5000, debug=False, use_reloader=False)
