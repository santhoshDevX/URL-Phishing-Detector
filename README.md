# 🛡️ PhishGuard — URL Phishing Detector

A complete end-to-end machine-learning web app that detects whether a URL is
**legitimate** or **phishing** in real time.

---

## 📁 Project Structure

```
phishing-detector/
│
├── app.py                  ← Flask backend (routes, model loading)
├── feature_extraction.py   ← URL → 15 numeric features
├── train_model.py          ← Dataset builder + ML training pipeline
├── requirements.txt        ← Python dependencies
│
├── model/                  ← Auto-created after training
│   ├── phishing_model.pkl
│   ├── scaler.pkl
│   └── meta.pkl
│
└── templates/
    └── index.html          ← Frontend UI (single file, no framework)
```

---

## ⚡ Quick Start

### 1. Create & activate a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

This creates `model/phishing_model.pkl`, `model/scaler.pkl`, and `model/meta.pkl`.

To use a **real CSV dataset** instead:

```bash
python train_model.py --csv path/to/dataset.csv
```

The CSV must have columns: `url` (string) and `label` (0 = legitimate, 1 = phishing).

**Recommended real datasets:**

| Dataset | URL |
|---------|-----|
| Kaggle Phishing Site URLs | https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls |
| UCI Phishing Websites | https://archive.ics.uci.edu/ml/datasets/phishing+websites |
| PhishTank | https://www.phishtank.com/developer_info.php |

### 4. Start the Flask server

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🔬 Extracted Features (15 total)

| # | Feature | Description |
|---|---------|-------------|
| 0 | `url_length` | Total characters in URL |
| 1 | `has_at_symbol` | `@` tricks browsers into ignoring the domain |
| 2 | `has_double_slash` | Redirect indicator (`//` after position 7) |
| 3 | `has_dash_in_domain` | Dashes in domain (e.g. `paypal-secure.xyz`) |
| 4 | `subdomain_count` | Dots in hostname − 1 |
| 5 | `https_flag` | 1 = HTTPS, 0 = HTTP |
| 6 | `has_ip_address` | Hostname is a raw IPv4 address |
| 7 | `suspicious_keywords` | Count of words like login, verify, bank, secure |
| 8 | `special_char_count` | Count of `$ % ! & = + # ~ ; , |` |
| 9 | `path_length` | Characters in URL path |
| 10 | `query_param_count` | Number of query parameters |
| 11 | `domain_length` | Characters in netloc |
| 12 | `digit_ratio` | Fraction of digits in URL |
| 13 | `dot_count` | Total dots |
| 14 | `has_port` | Non-standard port in URL |

---

## 🧪 Test URLs

| URL | Expected |
|-----|----------|
| `https://www.google.com` | ✅ Legitimate |
| `https://github.com/openai/gpt-4` | ✅ Legitimate |
| `http://paypal-login-secure.xyz/verify?account=1234` | 🚨 Phishing |
| `http://192.168.1.1/admin/login.php` | 🚨 Phishing |
| `http://apple-id-suspended.gq/unlock?id=user@email.com` | 🚨 Phishing |

---

## 🚨 Common Errors & Fixes

### `FileNotFoundError: Model not found`
**Cause:** You haven't trained the model yet.  
**Fix:** Run `python train_model.py`

### `ModuleNotFoundError: No module named 'flask'`
**Cause:** Virtual environment not active or deps not installed.  
**Fix:** `source venv/bin/activate && pip install -r requirements.txt`

### `Port 5000 already in use`
**Fix:** `python app.py` — or change port in `app.py`: `app.run(port=5001)`

### `Address already in use` on macOS
**Cause:** macOS AirPlay uses port 5000.  
**Fix:** Use port 5001 instead.

### Model accuracy is low
**Cause:** Synthetic dataset is small.  
**Fix:** Use a real CSV dataset with 10 000+ URLs.

---

## 🚀 Improvements & Next Steps

1. **Larger real dataset** — Use PhishTank or Kaggle (50k+ URLs) for better generalisation.
2. **WHOIS & DNS lookups** — Add domain age, registration country via `python-whois`.
3. **VirusTotal API integration** — Cross-check with threat intelligence.
4. **Browser extension** — See section below.
5. **Rate limiting** — Add `flask-limiter` to prevent abuse.
6. **Caching** — Cache recent predictions with `flask-caching` or Redis.
7. **Docker** — Containerise for easy deployment.
8. **CI/CD** — Re-train model weekly on fresh PhishTank data.

---

## 🧩 Browser Extension (Concept)

Converting this into a Chrome/Firefox extension involves three files:

```
phishguard-extension/
├── manifest.json       ← Extension config (MV3)
├── background.js       ← Service worker: intercepts navigation
└── popup.html          ← Small UI shown on toolbar click
```

**manifest.json** (key parts):
```json
{
  "manifest_version": 3,
  "name": "PhishGuard",
  "permissions": ["activeTab", "storage"],
  "host_permissions": ["http://localhost:5000/*"],
  "background": { "service_worker": "background.js" },
  "action": { "default_popup": "popup.html" }
}
```

**background.js** — on every navigation, send the URL to your Flask API:
```javascript
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'loading' && tab.url) {
    fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: tab.url })
    })
    .then(r => r.json())
    .then(data => {
      if (data.is_phishing) {
        chrome.notifications.create({
          type: 'basic',
          title: '🚨 PhishGuard Alert',
          message: `Phishing detected! (${data.confidence}% confidence)`,
          iconUrl: 'icon.png'
        });
      }
    });
  }
});
```

For production, deploy the Flask API to a cloud service (Railway, Render, Fly.io)
and update the URL in `background.js`.

---

## 📄 License

MIT — for educational and research purposes.
