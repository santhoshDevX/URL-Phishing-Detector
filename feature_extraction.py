"""
feature_extraction.py
---------------------
Extracts numerical features from a raw URL string.
These features are fed directly into the trained ML model.
"""

import re
import urllib.parse


# ── Suspicious keyword list ──────────────────────────────────────────────────
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "bank", "secure", "account", "update",
    "confirm", "signin", "password", "paypal", "ebay", "amazon",
    "apple", "microsoft", "support", "billing", "urgent", "suspend",
]


def extract_features(url: str) -> list:
    """
    Given a URL string, return a list of 15 numeric features.

    Feature index map
    -----------------
    0  url_length           – total character count
    1  has_at_symbol        – 1 if '@' present (classic phishing trick)
    2  has_double_slash     – 1 if '//' appears after the 7th char
    3  has_dash_in_domain   – 1 if '-' in the domain part
    4  subdomain_count      – number of dots in the hostname
    5  https_flag           – 1 if scheme is https
    6  has_ip_address       – 1 if hostname looks like an IPv4 address
    7  suspicious_keywords  – count of suspicious words found in URL
    8  special_char_count   – count of: $ % ! & = + # ~ ; , |
    9  path_length          – length of the URL path component
    10 query_param_count    – number of query parameters
    11 domain_length        – length of the netloc (domain) part
    12 digit_ratio          – proportion of digits in the full URL
    13 dot_count            – total dots in the URL
    14 has_port             – 1 if a non-standard port is specified
    """

    url = url.strip()
    features = []

    # ── Parse URL ────────────────────────────────────────────────────────────
    try:
        parsed = urllib.parse.urlparse(url if "://" in url else "http://" + url)
    except Exception:
        return [0] * 15

    full_url   = url.lower()
    hostname   = parsed.hostname or ""
    path       = parsed.path or ""
    query      = parsed.query or ""
    scheme     = parsed.scheme or ""

    # 0 – URL length
    features.append(len(url))

    # 1 – '@' symbol in URL (redirects browser to the part after '@')
    features.append(1 if "@" in url else 0)

    # 2 – '//' in path (potential redirect)
    features.append(1 if "//" in url[7:] else 0)

    # 3 – Dash in domain (e.g. paypal-secure.xyz)
    features.append(1 if "-" in hostname else 0)

    # 4 – Subdomain count (dots in hostname minus 1)
    dot_in_host = hostname.count(".")
    features.append(max(0, dot_in_host - 1))

    # 5 – HTTPS flag
    features.append(1 if scheme == "https" else 0)

    # 6 – IP address as hostname
    ip_pattern = r"^\d{1,3}(\.\d{1,3}){3}$"
    features.append(1 if re.match(ip_pattern, hostname) else 0)

    # 7 – Suspicious keyword count
    kw_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in full_url)
    features.append(kw_count)

    # 8 – Special characters
    special_chars = set("$%!&=+#~;,|")
    features.append(sum(1 for c in url if c in special_chars))

    # 9 – Path length
    features.append(len(path))

    # 10 – Query parameter count
    params = urllib.parse.parse_qs(query)
    features.append(len(params))

    # 11 – Domain (netloc) length
    features.append(len(parsed.netloc))

    # 12 – Digit ratio in full URL
    digit_count = sum(1 for c in url if c.isdigit())
    features.append(round(digit_count / max(len(url), 1), 4))

    # 13 – Total dots in URL
    features.append(url.count("."))

    # 14 – Non-standard port present?
    standard_ports = {80, 443, 8080, 8443, None}
    try:
        port = parsed.port
    except ValueError:
        port = None
    features.append(0 if port in standard_ports else 1)

    return features


FEATURE_NAMES = [
    "url_length", "has_at_symbol", "has_double_slash", "has_dash_in_domain",
    "subdomain_count", "https_flag", "has_ip_address", "suspicious_keywords",
    "special_char_count", "path_length", "query_param_count", "domain_length",
    "digit_ratio", "dot_count", "has_port",
]


if __name__ == "__main__":
    # Quick smoke-test
    test_urls = [
        "https://www.google.com",
        "http://paypal-login-secure.xyz/verify?account=1234",
        "http://192.168.1.1/phish",
        "https://amazon.com/login",
    ]
    print(f"{'URL':<55} {'Features'}")
    print("-" * 100)
    for u in test_urls:
        print(f"{u:<55} {extract_features(u)}")
