import os
import requests
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==============================
# CONFIG
# ==============================

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

MODEL_NAME = "ProsusAI/finbert"

# ==============================
# LOAD FINBERT MODEL
# ==============================

print("Loading FinBERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

print("Model loaded successfully!")

# ==============================
# FUNCTIONS
# ==============================

def get_global_news():
    url = (
        "https://newsapi.org/v2/top-headlines?"
        "category=business&"
        "language=en&"
        "pageSize=5&"
        f"apiKey={NEWSAPI_KEY}"
    )

    response = requests.get(url)
    data = response.json()

    if data.get("status") != "ok":
        raise Exception(f"NewsAPI Error: {data}")

    return data["articles"]


def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1)
    probs = probs.detach().numpy()[0]

    labels = ["Negative", "Neutral", "Positive"]
    sentiment = labels[probs.argmax()]
    confidence = float(probs.max())

    return sentiment, confidence


def sentiment_to_score(label):
    if label == "Positive":
        return 1
    elif label == "Negative":
        return -1
    else:
        return 0


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    requests.post(url, json=payload)


# ==============================
# MAIN LOGIC
# ==============================

def main():
    print("Fetching news...")
    articles = get_global_news()

    message = "🌍 *Insight Pasar Global*\n\n"
    total_score = 0

    for i, article in enumerate(articles, start=1):
        title = article["title"]
        description = article.get("description") or ""
        url = article["url"]

        text_for_analysis = title + ". " + description

        sentiment, confidence = analyze_sentiment(text_for_analysis)
        score = sentiment_to_score(sentiment)
        total_score += score

        emoji = "🟢" if sentiment == "Positive" else "🔴" if sentiment == "Negative" else "⚖️"

        message += (
            f"{i}. 💬 {title}\n"
            f"📊 Sentimen: {emoji} {sentiment} ({confidence:.2f})\n"
            f"🔗 {url}\n\n"
        )

    # ==============================
    # AGGREGATE SUMMARY
    # ==============================

    if total_score > 0:
        overall = "🟢 Bullish"
    elif total_score < 0:
        overall = "🔴 Bearish"
    else:
        overall = "⚖️ Netral"

    message += "────────────────────\n"
    message += f"📊 *Ringkasan Global*\n"
    message += f"Sentimen Hari Ini: {overall}\n"
    message += f"Skor Agregat: {total_score}\n"

    print(message)
    send_telegram(message)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        raise