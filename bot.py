import os
import requests
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

MODEL_NAME = "ProsusAI/finbert"

print("Memuat model analisis pasar...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
print("Model siap digunakan!")

# ==============================
# KATA KUNCI PASAR
# ==============================

KATA_POSITIF = [
    "rally", "surge", "jump", "soar", "beat",
    "strong", "record high", "growth", "gain", "optimistic"
]

KATA_NEGATIF = [
    "slump", "drop", "plunge", "miss",
    "weak", "loss", "decline", "recession", "fall", "fear"
]


def ambil_berita_global():
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
        raise Exception(data)

    return data["articles"]


def analisa_sentimen(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1)[0].detach().numpy()
    labels = ["Negative", "Neutral", "Positive"]

    sentimen = labels[probs.argmax()]
    confidence = float(probs.max())

    return sentimen, confidence


def boost_kata_kunci(text):
    text_lower = text.lower()
    boost = 0

    for kata in KATA_POSITIF:
        if kata in text_lower:
            boost += 0.3

    for kata in KATA_NEGATIF:
        if kata in text_lower:
            boost -= 0.3

    return boost


def skor_dasar(label, confidence):
    if label == "Positive":
        return 1 * confidence
    elif label == "Negative":
        return -1 * confidence
    else:
        return 0


def kirim_telegram(pesan):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": pesan,
        "parse_mode": "Markdown"
    }

    requests.post(url, json=payload)


def main():
    articles = ambil_berita_global()

    pesan = "🌍 *Ringkasan Sentimen Pasar Global*\n\n"
    total_score = 0

    for i, article in enumerate(articles, start=1):
        title = article["title"]
        description = article.get("description") or ""
        url = article["url"]

        text = title + ". " + description

        sentimen, confidence = analisa_sentimen(text)
        base_score = skor_dasar(sentimen, confidence)
        boost = boost_kata_kunci(text)
        final_score = base_score + boost

        total_score += final_score

        if final_score > 0.3:
            emoji = "🟢"
            label_id = "Menguat"
        elif final_score < -0.3:
            emoji = "🔴"
            label_id = "Melemah"
        else:
            emoji = "⚖️"
            label_id = "Netral"

        pesan += (
            f"{i}. 💬 {title}\n"
            f"📊 Arah Sentimen: {emoji} {label_id}\n"
            f"📈 Skor: {final_score:.2f}\n"
            f"🔗 {url}\n\n"
        )

    # ==============================
    # INDEKS PASAR
    # ==============================

    rata_rata = total_score / len(articles)
    indeks_pasar = int(rata_rata * 100)

    if indeks_pasar > 20:
        kondisi = "🟢 Pasar Global Cenderung Menguat"
    elif indeks_pasar < -20:
        kondisi = "🔴 Pasar Global Cenderung Melemah"
    else:
        kondisi = "⚖️ Pasar Global Cenderung Sideways"

    pesan += "───────────────\n"
    pesan += f"📊 *Indeks Panas Pasar: {indeks_pasar}*\n"
    pesan += f"{kondisi}\n"

    print(pesan)
    kirim_telegram(pesan)


if __name__ == "__main__":
    main()