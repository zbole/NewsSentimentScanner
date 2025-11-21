import feedparser
import requests
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from urllib.parse import quote
import time

from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime

# --- 1. 配置区域 ---
PUSH_TOKENS = [
    'eb50327c511447de8ec7b624d8d13c53',  # 第一个人的 Token
    '你的_第二个_TOKEN_粘贴在这里'           # 第二个人的 Token
]

KEYWORDS = ["TSLA", "NVDA", "AAPL", "AMD", "GOOG", "GOOGL"]
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

    # --- 1. 计算标题显示的中文方向 ---
    direction = ""  # 默认值
    emoji = "😐"

    if sentiment == "Positive":
        direction = "利好🔥"
        emoji = "🚀"
    elif sentiment == "Negative":
        direction = "利空❄️"
        emoji = "🔻"

    # --- 2. 准备正文内容 ---
    content = (
        f"### {emoji} {symbol} {direction}信号\n"
        f"- **情绪**: {sentiment}\n"
        f"- **置信度**: {confidence:.2f}\n"
        f"- **标题**: {title}\n"
        f"- **链接**: [点击查看]({link})"
    )

    # --- 3. 循环发送 ---
    for token in PUSH_TOKENS:
        data = {
            "token": token,
            # [修改点] 这里把原来的 "情绪异动" 改成了动态变量
            "title": f"{symbol} 出现{direction} ({confidence:.2f})",
            "content": content,
            "template": "markdown"
        }
        try:
            response = requests.post(url, json=data)
            # 检查一下响应状态
            resp_json = response.json()
            if resp_json.get('code') == 200:
                print(f"--> 已成功推送到 Token: ...{token[-4:]}")
            else:
                print(f"--> 推送失败 (Token: ...{token[-4:]}): {resp_json.get('msg')}")
        except Exception as e:
            print(f"网络请求错误: {e}")


def main():
    print(f"Github Action 启动: 开始监控 {KEYWORDS}")
    print("-" * 30)

    # 设定一个时间窗口：只看最近 4 小时的新闻 (避免重复推送老旧新闻)
    # 因为 GitHub Actions 每次运行都是“失忆”的，所以必须靠时间来过滤
    time_threshold = datetime.now() - timedelta(minutes=40)

    for query in KEYWORDS:
        print(f"正在扫描: {query} ...")
        try:
            articles = fetch_news(query)

            for item in articles:
                title = item.title
                link = item.link

                # --- 新增：时间过滤逻辑 ---
                # Google News RSS 的时间格式比较复杂，用 parsedate_to_datetime 解析
                try:
                    pub_date = parsedate_to_datetime(item.published)
                    # 把 pub_date 转成不带时区的 timestamp 进行比较，或者直接忽略时区
                    if pub_date.replace(tzinfo=None) < time_threshold:
                        print(f"  [跳过] 新闻太旧: {title[:15]}...")
                        continue
                except Exception as e:
                    print(f"  时间解析失败，默认处理: {e}")

                # AI 分析
                sentiment, confidence = analyze_sentiment_finbert(title)

                print(f"[{query}] {sentiment} ({confidence:.2f}) - {title[:30]}...")

                # 过滤策略：不是中性 且 置信度够高
                if sentiment != 'Neutral' and confidence > CONFIDENCE_THRESHOLD:
                    send_wechat_alert(query, title, sentiment, confidence, link)

        except Exception as e:
            print(f"抓取错误: {e}")

    print("本次扫描结束，脚本自动退出 (等待下一次 Cron 唤醒)")


if __name__ == "__main__":
    main()
