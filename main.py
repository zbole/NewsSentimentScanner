import feedparser
import requests
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from urllib.parse import quote
import time

# --- 1. 配置区域 ---
PUSH_TOKENS = [
    'eb50327c511447de8ec7b624d8d13c53',  # 第一个人的 Token
    '你的_第二个_TOKEN_粘贴在这里'           # 第二个人的 Token
]

KEYWORDS = ["TSLA", "NVDA", "AAPL", "AMD"]
CONFIDENCE_THRESHOLD = 0.85

# --- 2. 加载 FinBERT 模型 (第一次运行会下载) ---
print("正在加载 FinBERT AI 模型，请稍候...")
# 使用专门针对金融情绪微调过的模型
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
labels = ['Neutral', 'Positive', 'Negative']  # 注意：这个模型的标签顺序通常是这样，但也可能是 [Neutral, Positive, Negative]，下面逻辑已适配


# --- 3. 核心功能函数 ---

def fetch_news(query):
    """从 Google News RSS 获取新闻"""
    encoded_query = quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    return feed.entries[:5]  # 每个关键词只取最新的 5 条


def analyze_sentiment_finbert(text):
    """使用 FinBERT 进行 AI 分析"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).numpy()[0]
    max_index = np.argmax(probabilities)

    sentiment = labels[max_index]
    confidence = probabilities[max_index]

    return sentiment, confidence


def send_wechat_alert(symbol, title, sentiment, confidence, link):
    """发送微信推送 (支持多账号)"""
    url = 'http://www.pushplus.plus/send'

    # 根据情绪换个 Emoji
    emoji = "😐"
    if sentiment == "Positive": emoji = "🚀 利好"
    if sentiment == "Negative": emoji = "🔻 利空"

    content = (
        f"### {emoji} {symbol} 信号触发\n"
        f"- **情绪**: {sentiment}\n"
        f"- **置信度**: {confidence:.2f}\n"
        f"- **标题**: {title}\n"
        f"- **链接**: [点击查看]({link})"
    )

    # --- 修改点：循环发送给列表里的每一个 Token ---
    for token in PUSH_TOKENS:
        data = {
            "token": token,
            "title": f"{symbol} 情绪异动",
            "content": content,
            "template": "markdown"
        }
        try:
            response = requests.post(url, json=data)
            # 检查一下响应状态，防止 token 填错
            resp_json = response.json()
            if resp_json.get('code') == 200:
                print(f"--> 已成功推送到 Token: ...{token[-4:]}")
            else:
                print(f"--> 推送失败 (Token: ...{token[-4:]}): {resp_json.get('msg')}")
        except Exception as e:
            print(f"网络请求错误: {e}")


# --- 4. 主程序 ---
def main():
    print(f"开始监控以下目标: {KEYWORDS}")
    print("-" * 30)

    # 用于去重，防止同一条新闻重复推送
    seen_links = set()

    while True:
        for query in KEYWORDS:
            print(f"正在扫描: {query} ...")
            try:
                articles = fetch_news(query)

                for item in articles:
                    link = item.link
                    title = item.title

                    # 如果这条新闻已经推过了，跳过
                    if link in seen_links:
                        continue
                    seen_links.add(link)

                    # AI 分析
                    sentiment, confidence = analyze_sentiment_finbert(title)

                    print(f"[{query}] {sentiment} ({confidence:.2f}) - {title[:30]}...")

                    # 过滤策略：不是中性 且 置信度够高
                    if sentiment != 'Neutral' and confidence > CONFIDENCE_THRESHOLD:
                        send_wechat_alert(query, title, sentiment, confidence, link)

            except Exception as e:
                print(f"抓取错误: {e}")

        print("休息 5 分钟...")
        time.sleep(300)  # 300秒 = 5分钟扫描一次


if __name__ == "__main__":
    main()