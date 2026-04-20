"""
Fake News Detector
==================
Uses TF-IDF + Logistic Regression on the LIAR / WELFake dataset.
Run: pip install pandas scikit-learn gradio requests
Then: python fake_news_detector.py
"""

import pandas as pd
import numpy as np
import re
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# ── 1. Load / create dataset ──────────────────────────────────────────────────

def load_data():
    """
    Try to load the WELFake dataset (CSV). If not found, use a small
    built-in demo set so the script always runs out-of-the-box.

    To get full accuracy, download WELFake_Dataset.csv from:
    https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
    and place it in the same folder as this script.
    """
    csv_path = "WELFake_Dataset.csv"

    if os.path.exists(csv_path):
        print("Loading WELFake dataset...")
        df = pd.read_csv(csv_path)
        # WELFake columns: Unnamed: 0, title, text, label (0=fake, 1=real)
        df = df[["title", "text", "label"]].dropna()
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
        print(f"Loaded {len(df):,} articles.")
    else:
        print("WELFake dataset not found — using built-in demo data.")
        print("For best results, download WELFake_Dataset.csv from Kaggle.\n")
        demo = [
            ("Breaking: Scientists confirm moon landing was staged in Hollywood", 0),
            ("Federal Reserve raises interest rates by 0.25 basis points", 1),
            ("SHOCKING: 5G towers secretly injecting nanobots via radio waves", 0),
            ("WHO reports global measles cases rising for third consecutive year", 1),
            ("Obama signs secret executive order banning the Bible nationwide", 0),
            ("Senate passes bipartisan infrastructure bill 69-30", 1),
            ("Doctors HATE this one weird trick that cures all cancers overnight", 0),
            ("New study links ultra-processed foods to increased dementia risk", 1),
            ("George Soros admits funding antifa to destroy America in leaked video", 0),
            ("NASA's James Webb telescope captures deepest infrared image of universe", 1),
            ("Vaccines contain microchips that track your location via Bluetooth", 0),
            ("IMF lowers global growth forecast amid persistent inflation pressures", 1),
            ("Hillary Clinton arrested, taken to Guantanamo Bay by military tribunal", 0),
            ("SpaceX Starship completes first full orbital test flight successfully", 1),
            ("Leaked Pfizer documents reveal vaccine turns recipients into zombies", 0),
            ("European Central Bank announces quantitative tightening program", 1),
            ("FEMA camps being secretly built to imprison Trump supporters", 0),
            ("UN report warns climate change could displace 1 billion people by 2050", 1),
            ("Secret underground tunnel network connects all major US cities for elites", 0),
            ("Apple reports record quarterly revenue of 97.3 billion dollars", 1),
        ] * 50  # repeat to simulate larger dataset for demo
        df = pd.DataFrame(demo, columns=["content", "label"])

    return df



def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)       # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)             # keep only letters
    text = re.sub(r"\s+", " ", text).strip()           # collapse whitespace
    return text

# ── 3. Build & train pipeline ─────────────────────────────────────────────────

def train_model(df):
    print("\nCleaning text...")
    df["clean"] = df["content"].apply(clean_text)

    X = df["clean"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),     # unigrams + bigrams
            sublinear_tf=True,      # log-scale TF
            min_df=2,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            solver="lbfgs",
            n_jobs=-1,
        )),
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    return pipeline

# ── 4. Save model ─────────────────────────────────────────────────────────────

def save_model(pipeline, path="model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved to {path}")

def load_model(path="model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

# ── 5. Predict function ───────────────────────────────────────────────────────

def predict(text, pipeline):
    cleaned = clean_text(text)
    prob = pipeline.predict_proba([cleaned])[0]
    label = pipeline.predict([cleaned])[0]
    return {
        "label": "REAL" if label == 1 else "FAKE",
        "confidence": float(max(prob)),
        "fake_prob": float(prob[0]),
        "real_prob": float(prob[1]),
    }

# ── 6. Gradio web UI ──────────────────────────────────────────────────────────

def launch_gradio(pipeline):
    try:
        import gradio as gr
    except ImportError:
        print("\nGradio not installed. Run: pip install gradio")
        print("Falling back to CLI mode.\n")
        cli_demo(pipeline)
        return

    def gradio_predict(text):
        if not text.strip():
            return "Please enter some text.", "", ""
        result = predict(text, pipeline)
        label = result["label"]
        conf  = result["confidence"] * 100
        fake_p = result["fake_prob"] * 100
        real_p = result["real_prob"] * 100
        verdict = f"{'🔴 FAKE NEWS' if label == 'FAKE' else '🟢 REAL NEWS'} — {conf:.1f}% confidence"
        breakdown = f"Fake: {fake_p:.1f}%  |  Real: {real_p:.1f}%"
        return verdict, breakdown

    iface = gr.Interface(
        fn=gradio_predict,
        inputs=gr.Textbox(
            lines=5,
            placeholder="Paste a news headline or article excerpt here...",
            label="News text",
        ),
        outputs=[
            gr.Textbox(label="Verdict"),
            gr.Textbox(label="Probability breakdown"),
        ],
        title="Fake News Detector",
        description=(
            "Enter a news headline or article text and the model will predict "
            "whether it is real or fake news. Built with TF-IDF + Logistic Regression."
        ),
        examples=[
            ["Scientists discover that drinking coffee extends life by 10 years, big pharma furious"],
            ["Federal Reserve holds interest rates steady at 5.25%, citing stable inflation data"],
            ["NASA confirms alien spacecraft found on dark side of the moon, cover-up revealed"],
        ],
        theme=gr.themes.Soft(),
    )

    print("\nLaunching Gradio demo — copy the URL to share!")
    iface.launch(share=True)

# ── 7. CLI fallback ───────────────────────────────────────────────────────────

def cli_demo(pipeline):
    print("\n=== Fake News Detector CLI ===")
    print("Type a headline or paste article text. Press Ctrl+C to quit.\n")
    while True:
        try:
            text = input("Enter news text: ").strip()
            if not text:
                continue
            r = predict(text, pipeline)
            print(f"  Result    : {r['label']}")
            print(f"  Confidence: {r['confidence']:.1%}")
            print(f"  Fake prob : {r['fake_prob']:.1%}")
            print(f"  Real prob : {r['real_prob']:.1%}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

# ── 8. Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    MODEL_PATH = "model.pkl"

    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}...")
        pipeline = load_model(MODEL_PATH)
    else:
        df = load_data()
        pipeline = train_model(df)
        save_model(pipeline, MODEL_PATH)

    launch_gradio(pipeline)
