"""
app.py
------
Flask backend for the URL Phishing Detector.

Routes
------
GET  /            → Serves the main HTML UI
POST /predict     → Accepts JSON {"url": "..."}, returns prediction JSON
GET  /health      → Health-check endpoint
"""

import os
import pickle
import logging
from flask import Flask, render_template, request, jsonify

from feature_extraction import extract_features, FEATURE_NAMES

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── App init ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Load model artifacts ─────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

def load_artifacts():
    """Load model, scaler, and metadata from disk."""
    model_path  = os.path.join(MODEL_DIR, "phishing_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    meta_path   = os.path.join(MODEL_DIR, "meta.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run  python train_model.py  first."
        )

    with open(model_path,  "rb") as f: model  = pickle.load(f)
    with open(scaler_path, "rb") as f: scaler = pickle.load(f)
    with open(meta_path,   "rb") as f: meta   = pickle.load(f)

    log.info("Model loaded: %s", meta.get("model_name", "unknown"))
    return model, scaler, meta

try:
    MODEL, SCALER, META = load_artifacts()
    MODEL_READY = True
except FileNotFoundError as e:
    log.warning(str(e))
    MODEL, SCALER, META = None, None, {}
    MODEL_READY = False


# ── Prediction logic ─────────────────────────────────────────────────────────
def predict_url(url: str) -> dict:
    """
    Extract features from url and run them through the classifier.

    Returns
    -------
    dict with keys:
        url          – original URL
        prediction   – "Phishing" or "Legitimate"
        confidence   – float 0-100 (percentage)
        risk_level   – "High" / "Medium" / "Low"
        features     – dict of extracted feature values
        is_phishing  – bool
    """
    features = extract_features(url)
    feature_dict = dict(zip(FEATURE_NAMES, features))

    import numpy as np
    X = np.array(features).reshape(1, -1)

    # Logistic Regression needs scaling; tree models do not
    model_name = META.get("model_name", "")
    if "Logistic" in model_name:
        X = SCALER.transform(X)

    # Probability scores
    proba = MODEL.predict_proba(X)[0]   # [P(legit), P(phishing)]
    phishing_prob = float(proba[1]) * 100
    label = MODEL.predict(X)[0]         # 0 or 1

    is_phishing = bool(label == 1)
    prediction  = "Phishing" if is_phishing else "Legitimate"

    if phishing_prob >= 70:
        risk = "High"
    elif phishing_prob >= 40:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "url":         url,
        "prediction":  prediction,
        "confidence":  round(phishing_prob if is_phishing else 100 - phishing_prob, 1),
        "risk_level":  risk,
        "features":    feature_dict,
        "is_phishing": is_phishing,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main UI."""
    return render_template("index.html", model_ready=MODEL_READY)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON: {"url": "https://example.com"}
    Returns JSON with prediction details.
    """
    if not MODEL_READY:
        return jsonify({
            "error": "Model not loaded. Run  python train_model.py  first."
        }), 503

    data = request.get_json(silent=True) or {}
    url  = (data.get("url") or "").strip()

    if not url:
        return jsonify({"error": "No URL provided."}), 400

    # Basic length guard
    if len(url) > 2048:
        return jsonify({"error": "URL too long (max 2048 chars)."}), 400

    try:
        result = predict_url(url)
        log.info("Prediction: %-12s | confidence=%.1f%% | %s",
                 result["prediction"], result["confidence"], url[:80])
        return jsonify(result), 200
    except Exception as exc:
        log.error("Prediction error: %s", exc)
        return jsonify({"error": "Internal prediction error.", "detail": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status":      "ok",
        "model_ready": MODEL_READY,
        "model_name":  META.get("model_name", "n/a"),
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
