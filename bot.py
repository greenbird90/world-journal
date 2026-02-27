import asyncio
import requests
import html
import json
import os
from datetime import datetime, timedelta
from telegram import Bot

# ==============================
# 🔧 KONFIGURASI (Ambil dari GitHub Secrets)
# ==============================
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
HF_API_KEY = os.getenv("HF_API_KEY")

TREND_FILE = "sentiment_trend.json"

# HuggingFace FinBERT endpoint
HF_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}


# ==============================
# 📰 AMBIL BERITA
# ==============================
def get_top_headlines():
    url = f'https://newsapi.org/v2/top-headlines?category=business&language=en&pageSize=10&apiKey={NEWSAPI_KEY}'
    response = requests.get(url)
    data = response.json()
    return data.get('articles', [])


# ==============================
# 🔎 FILTER RELEVANSI MARKET
# ==============================
def is_market_relevant(text):
    keywords = [
        'stock','market','shares','earnings','revenue',
        'inflation','interest rate','fed','bank',
        'oil','gold','acquisition','merger','ipo',
        'downgrade','upgrade','forecast','guidance',
        'economy','bond','treasury','nasdaq','s&p'
    ]
    text_lower = text.lower()
    return any(k in text_lower for k in keywords)


# ==============================
# 📊 ANALISIS SENTIMEN VIA API
# ==============================
def analyze_sentiment(text):
    payload = {"inputs": text[:512]}

    response = requests.post(HF_URL, headers=HF_HEADERS, json=payload)
    result = response.json()

    # Handle model loading case
    if isinstance(result, dict) and "error" in result:
        return "Netral", 0.5

    result = result[0]
    best = max(result, key=lambda x: x['score'])

    label = best['label'].lower()
    score = best['score']

    if score < 0.60:
        return "Netral", score

    if label == "positive":
        return "Positif", score
    elif label == "negative":
        return "Negatif", score
    else:
        return "Netral", score


# ==============================
# 💡 GENERATE INSIGHT
# ==============================
def generate_insight(text, sentiment):
    text_lower = text.lower()

    if 'inflation' in text_lower:
        return "💡 Inflasi mempengaruhi arah suku bunga dan daya beli."

    if 'interest rate' in text_lower or 'fed' in text_lower:
        return "💡 Kebijakan suku bunga berdampak pada valuasi saham dan arus modal global."

    if 'earnings' in text_lower or 'revenue' in text_lower:
        return "💡 Laporan kinerja perusahaan sering menjadi katalis utama harga saham."

    if 'oil' in text_lower:
        return "💡 Harga minyak mempengaruhi sektor energi dan tekanan inflasi."

    if sentiment == "Positif":
        return "💡 Sentimen mendukung risk appetite jangka pendek."

    if sentiment == "Negatif":
        return "💡 Tekanan sentimen dapat memicu aksi defensif."

    return "💡 Berita bersifat informatif, perlu konfirmasi lanjutan."


# ==============================
# 📈 TREND 5 HARI
# ==============================
def update_sentiment_trend(today_sentiment):
    data = {}

    if os.path.exists(TREND_FILE):
        with open(TREND_FILE, "r") as f:
            data = json.load(f)

    today_str = datetime.now().strftime("%Y-%m-%d")
    data[today_str] = today_sentiment

    cutoff = datetime.now() - timedelta(days=5)
    data = {
        k: v for k, v in data.items()
        if datetime.strptime(k, "%Y-%m-%d") >= cutoff
    }

    with open(TREND_FILE, "w") as f:
        json.dump(data, f, indent=2)

    values = list(data.values())
    if len(values) < 2:
        return "📊 Tren belum cukup data."

    if values[-1] > values[0]:
        return "🔼 Sentimen membaik."
    elif values[-1] < values[0]:
        return "🔽 Sentimen melemah."
    else:
        return "➡️ Sentimen stabil."


# ==============================
# ✉️ TELEGRAM
# ==============================
async def send_to_telegram(message):
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=message,
        parse_mode='HTML',
        disable_web_page_preview=True
    )


# ==============================
# 🚀 MAIN
# ==============================
async def main():
    articles = get_top_headlines()

    if not articles:
        await send_to_telegram("Tidak ada berita bisnis terbaru.")
        return

    message = "<b>📈 Insight Pasar Global</b>\n\n"

    daily_score = 0
    count = 0

    for article in articles:
        title = article.get('title') or ''
        description = article.get('description') or ''
        url = article.get('url', '')

        content = f"{title}. {description}"

        if not is_market_relevant(content):
            continue

        count += 1

        sentiment, confidence = analyze_sentiment(content)

        if sentiment == "Positif":
            daily_score += confidence
        elif sentiment == "Negatif":
            daily_score -= confidence

        message += (
            f"{count}. 💬 <b>{html.escape(title)}</b>\n"
            f"📊 Sentimen: {sentiment} ({confidence:.2f})\n"
            f"📝 {html.escape(description)}\n"
            f"{generate_insight(content, sentiment)}\n"
            f"🔗 {url}\n\n"
        )

    if daily_score > 1:
        global_sentiment = "📈 <b>Bias Positif</b>"
        today_sentiment = 1
    elif daily_score < -1:
        global_sentiment = "📉 <b>Bias Negatif</b>"
        today_sentiment = -1
    else:
        global_sentiment = "⚖️ <b>Netral</b>"
        today_sentiment = 0

    trend = update_sentiment_trend(today_sentiment)

    message += (
        "<b>────────────────────────</b>\n"
        "<b>📊 Ringkasan Global</b>\n"
        f"🔸 Sentimen Hari Ini: {global_sentiment}\n"
        f"🔢 Skor Agregat: {daily_score:.2f}\n\n"
        f"{trend}\n\n"
        "💬 <i>Bukan sinyal trading langsung.</i>"
    )

    if count == 0:
        message = "Tidak ada berita relevan pasar hari ini."

    await send_to_telegram(message)


if __name__ == "__main__":
    asyncio.run(main())
