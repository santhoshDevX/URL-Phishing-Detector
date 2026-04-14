"""
train_model.py
--------------
1. Generates a realistic synthetic dataset (or loads a real CSV if provided).
2. Extracts features from every URL.
3. Trains a Random Forest classifier.
4. Evaluates accuracy and saves the model + scaler to /model/.

Usage
-----
    python train_model.py

To use a real CSV dataset instead of the synthetic one:
    python train_model.py --csv path/to/your/dataset.csv

Real dataset sources
--------------------
- UCI ML Phishing Websites: https://archive.ics.uci.edu/ml/datasets/phishing+websites
- Kaggle Phishing URL dataset: https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls
  CSV must have columns:  url, label   (label: 0=legitimate, 1=phishing)
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

from feature_extraction import extract_features, FEATURE_NAMES

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Synthetic dataset ────────────────────────────────────────────────────────
LEGITIMATE_URLS = [
    "https://www.google.com",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://github.com/openai/gpt-4",
    "https://stackoverflow.com/questions/tagged/python",
    "https://www.wikipedia.org/wiki/Machine_learning",
    "https://www.amazon.com/dp/B08N5WRWNW",
    "https://www.reddit.com/r/MachineLearning",
    "https://docs.python.org/3/library/urllib.html",
    "https://scikit-learn.org/stable/modules/ensemble.html",
    "https://www.bbc.com/news/technology",
    "https://www.nytimes.com/section/technology",
    "https://www.linkedin.com/in/someprofile",
    "https://twitter.com/OpenAI",
    "https://www.microsoft.com/en-us/microsoft-365",
    "https://www.apple.com/iphone",
    "https://www.coursera.org/learn/machine-learning",
    "https://www.udemy.com/course/python-bootcamp",
    "https://flask.palletsprojects.com/en/3.0.x/",
    "https://pandas.pydata.org/docs/",
    "https://numpy.org/doc/stable/",
    "https://www.kaggle.com/competitions",
    "https://arxiv.org/abs/1706.03762",
    "https://www.cloudflare.com/en-gb/learning/",
    "https://developer.mozilla.org/en-US/docs/Web",
    "https://www.w3schools.com/html/",
    "https://www.shopify.com/blog/ecommerce",
    "https://www.netflix.com/browse",
    "https://www.spotify.com/us/premium/",
    "https://www.dropbox.com/features",
    "https://www.notion.so/product",
    "https://trello.com/b/nC8QJJoZ/agile-sprint-board",
    "https://slack.com/intl/en-gb/features",
    "https://zoom.us/pricing",
    "https://www.atlassian.com/software/jira",
    "https://www.postgresql.org/docs/",
    "https://www.mongodb.com/docs/",
    "https://reactjs.org/docs/getting-started.html",
    "https://vuejs.org/guide/introduction.html",
    "https://angular.io/docs",
    "https://tailwindcss.com/docs/installation",
    "https://www.paypal.com/us/home",
    "https://www.ebay.com/sch/i.html?_nkw=laptop",
    "https://www.etsy.com/market/handmade_gifts",
    "https://www.alibaba.com/trade/search",
    "https://www.booking.com/flights",
    "https://www.airbnb.com/rooms/12345",
    "https://www.tripadvisor.com/Hotels",
    "https://www.healthline.com/nutrition",
    "https://www.webmd.com/cold-and-flu/default.htm",
    "https://www.investopedia.com/terms/s/stockmarket.asp",
]

PHISHING_URLS = [
    "http://paypal-login-secure.xyz/verify?account=1234&token=abcdef",
    "http://192.168.1.1/admin/login.php",
    "http://amazon-security-alert.tk/signin",
    "http://secure-banking-update.ml/account/verify",
    "http://apple-id-suspended.gq/unlock?id=user@email.com",
    "http://microsoft-support-urgent.cf/update",
    "http://ebay-login-confirm.pw/signin?redirect=account",
    "http://netflix-billing-update.xyz/payment",
    "http://dropbox-file-share.tk/download?file=invoice.pdf",
    "http://login.bank-of-america-secure.gq/account",
    "http://verify.paypal-accounts-suspended.ml/",
    "http://google-account-recovery.tk/verify?email=victim",
    "http://update.steam-community-items.xyz/trade",
    "http://facebook-login-help.tk/account/recover",
    "http://twitter-security-verify.gq/2fa/reset",
    "http://support-apple.com-id-locked.tk/signin",
    "http://signin.amazon-prime-membership.cf/renew",
    "http://account.instagram-verify.ml/checkpoint",
    "http://mail.google-account-update.gq/signin",
    "http://secure.bitcoin-wallet-restore.xyz/recover",
    "http://10.0.0.1/phishing/page.html",
    "http://172.16.254.1/login?redirect=bank",
    "http://your-bank-login.com-secure-update.tk/",
    "http://lloyds-bank-online-secure.gq/login",
    "http://hsbc-verify-account-secure.ml/signin",
    "http://citibank-alert-account-update.xyz/",
    "http://chase-bank-secure-login.tk/verify",
    "http://wellsfargo-account-alert.cf/update",
    "http://barclays-online-secure.gq/login",
    "http://halifax-account-update-secure.ml/",
    "http://irs-tax-refund-pending.xyz/claim",
    "http://fedex-package-tracking-update.tk/delivery",
    "http://dhl-shipment-notification.gq/track",
    "http://ups-delivery-confirm.ml/schedule",
    "http://usps-package-held-secure.xyz/verify",
    "http://covid-relief-fund-apply.tk/register",
    "http://nhs-covid-vaccine-booking.gq/schedule",
    "http://stimulus-check-claim.ml/apply?ssn=",
    "http://unemployment-benefits-update.xyz/verify",
    "http://crypto-doubler-investment.tk/send",
    "http://binance-withdraw-verification.gq/confirm",
    "http://coinbase-account-review.ml/verify",
    "http://metamask-restore-wallet.xyz/seed",
    "http://opensea-nft-airdrop.tk/claim?wallet=",
    "http://roblox-free-robux.gq/generate",
    "http://fortnite-vbucks-generator.ml/free",
    "http://steam-free-games.xyz/redeem?code=",
    "http://adobe-license-expired.tk/renew",
    "http://norton-antivirus-renew-urgent.gq/pay",
    "http://mcafee-subscription-expired.ml/extend",
]


def build_synthetic_dataset() -> pd.DataFrame:
    """Build a balanced synthetic dataset with realistic URL examples."""
    records = []
    for url in LEGITIMATE_URLS:
        records.append({"url": url, "label": 0})
    for url in PHISHING_URLS:
        records.append({"url": url, "label": 1})

    # Augment legitimate URLs (add common paths/params to boost variety)
    aug_legit = []
    paths = ["/about", "/contact", "/products", "/services/details", "/help/faq"]
    for base_url in LEGITIMATE_URLS[:20]:
        for p in paths:
            aug_legit.append({"url": base_url.rstrip("/") + p, "label": 0})

    # Augment phishing URLs
    aug_phish = []
    suffixes = ["?redirect=true", "&token=xyz123", "/verify/step2", "?id=9999&confirm=1"]
    for base_url in PHISHING_URLS[:20]:
        for s in suffixes:
            aug_phish.append({"url": base_url + s, "label": 1})

    all_records = records + aug_legit + aug_phish
    df = pd.DataFrame(all_records).drop_duplicates(subset="url").reset_index(drop=True)

    print(f"  Synthetic dataset: {len(df)} rows  "
          f"(legit={len(df[df.label==0])}, phishing={len(df[df.label==1])})")
    return df


def load_real_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load a real CSV dataset.
    Expected columns: 'url' (string) and 'label' (0=legit, 1=phishing).
    Handles common label formats automatically.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    # Normalise common label column names
    for col in df.columns:
        if col in ("type", "status", "result", "class", "target"):
            df = df.rename(columns={col: "label"})
            break

    # Normalise label values
    label_map = {
        "phishing": 1, "malicious": 1, "bad": 1, "1": 1, 1: 1,
        "legitimate": 0, "benign": 0, "good": 0, "safe": 0, "0": 0, 0: 0,
    }
    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["url", "label"])
    df["label"] = df["label"].astype(int)

    print(f"  Loaded CSV: {len(df)} rows  "
          f"(legit={len(df[df.label==0])}, phishing={len(df[df.label==1])})")
    return df[["url", "label"]]


def build_feature_matrix(df: pd.DataFrame):
    """Extract features for every URL row."""
    print("  Extracting features…")
    X = np.array([extract_features(u) for u in df["url"]])
    y = df["label"].values
    return X, y


def train_and_evaluate(X_train, X_test, y_train, y_test, scaler):
    """Train multiple models, pick the best, return it."""
    models = {
        "RandomForest":        RandomForestClassifier(
                                   n_estimators=200, max_depth=12,
                                   min_samples_split=5, random_state=RANDOM_STATE,
                                   class_weight="balanced"),
        "GradientBoosting":    GradientBoostingClassifier(
                                   n_estimators=150, learning_rate=0.1,
                                   max_depth=5, random_state=RANDOM_STATE),
        "LogisticRegression":  LogisticRegression(
                                   max_iter=1000, C=1.0,
                                   class_weight="balanced",
                                   random_state=RANDOM_STATE),
    }

    best_model, best_acc, best_name = None, 0.0, ""

    for name, model in models.items():
        # Logistic Regression benefits from scaling
        X_tr = scaler.transform(X_train) if name == "LogisticRegression" else X_train
        X_te = scaler.transform(X_test)  if name == "LogisticRegression" else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_test, y_pred)

        # 5-fold CV on training data
        cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring="accuracy")

        print(f"\n  ── {name} ──")
        print(f"     Test Accuracy : {acc:.4f}")
        print(f"     CV  Accuracy  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"     Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        print(classification_report(y_test, y_pred,
                                    target_names=["Legitimate", "Phishing"]))

        if acc > best_acc:
            best_acc, best_model, best_name = acc, model, name

    print(f"\n  ✅ Best model: {best_name}  (accuracy={best_acc:.4f})")
    return best_model, best_name


def save_artifacts(model, scaler, model_name: str):
    """Persist model + scaler to disk."""
    model_path  = os.path.join(MODEL_DIR, "phishing_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    meta_path   = os.path.join(MODEL_DIR, "meta.pkl")

    with open(model_path,  "wb") as f: pickle.dump(model,  f)
    with open(scaler_path, "wb") as f: pickle.dump(scaler, f)
    with open(meta_path,   "wb") as f:
        pickle.dump({"model_name": model_name, "features": FEATURE_NAMES}, f)

    print(f"\n  💾 Saved → {model_path}")
    print(f"  💾 Saved → {scaler_path}")
    print(f"  💾 Saved → {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Train phishing URL detector")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to real CSV dataset (columns: url, label)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  URL PHISHING DETECTOR — Model Training")
    print("="*60)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("\n[1] Loading dataset…")
    df = load_real_dataset(args.csv) if args.csv else build_synthetic_dataset()

    # ── 2. Feature extraction ─────────────────────────────────────────────────
    print("\n[2] Building feature matrix…")
    X, y = build_feature_matrix(df)
    print(f"     X shape: {X.shape}  |  class balance: "
          f"legit={int((y==0).sum())}, phishing={int((y==1).sum())}")

    # ── 3. Train / test split ─────────────────────────────────────────────────
    print("\n[3] Splitting data (80 / 20)…")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # ── 4. Scaler (fit on train only) ─────────────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(X_train)

    # ── 5. Train & evaluate ───────────────────────────────────────────────────
    print("\n[4] Training models…")
    best_model, best_name = train_and_evaluate(
        X_train, X_test, y_train, y_test, scaler
    )

    # ── 6. Feature importances (tree-based models) ────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        print("\n[5] Feature importances:")
        pairs = sorted(zip(FEATURE_NAMES, best_model.feature_importances_),
                       key=lambda x: x[1], reverse=True)
        for name, imp in pairs:
            bar = "█" * int(imp * 50)
            print(f"     {name:<25} {imp:.4f}  {bar}")

    # ── 7. Save ───────────────────────────────────────────────────────────────
    print("\n[6] Saving artifacts…")
    save_artifacts(best_model, scaler, best_name)

    print("\n✅ Training complete!\n")


if __name__ == "__main__":
    main()
