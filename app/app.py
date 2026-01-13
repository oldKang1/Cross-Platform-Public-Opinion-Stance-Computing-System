from flask import Flask, send_from_directory, request, jsonify
import os
import re
import csv
import shutil
import threading
import subprocess
import pandas as pd
from collections import Counter

import comments_nlp

app = Flask(__name__)


crawl_data = {
    "is_running": False,
    "progress": 0,
    "total_comments": 0,
    "video_url": "",
    "error": None,
    "message": ""
}

nlp_data = {
    "is_running": False,
    "progress": 0,
    "keywords": [],
    "sentiment": {"pos": 0.0, "neu": 0.0, "neg": 0.0},
    "timeline": [],
    "error": None,
    "message": ""
}


APP_DIR = os.path.dirname(os.path.abspath(__file__))


DATA_CANDIDATES = [
    os.path.join(os.path.dirname(APP_DIR), "data"),
    os.path.join(APP_DIR, "data"),
]

DATA_DIR = next((p for p in DATA_CANDIDATES if os.path.isdir(p)), None)
if DATA_DIR is None:
    
    raise FileNotFoundError(f"找不到 data 目录，已尝试: {DATA_CANDIDATES}")

CRAWLER_SCRIPT = os.path.join(DATA_DIR, "blbl.py")
CRAWLER_OUT_CSV = os.path.join(DATA_DIR, "bili_comments.csv")

NLP_SCRIPT = os.path.join(DATA_DIR, "comments_nlp.py")
NLP_IN_CSV = os.path.join(DATA_DIR, "comments1.csv")
NLP_OUT_CSV = os.path.join(DATA_DIR, "comments_nlp.csv")


@app.route("/")
def home():
    return send_from_directory("templates", "index.html")

@app.route("/show")
def show():
    return send_from_directory("templates", "show.html")

@app.route("/blbl")
def blbl():
    return send_from_directory("templates", "blbl.html")


def _modify_script_to_use_passed_url(script_path: str) -> None:
    with open(script_path, "r", encoding="utf-8") as f:
        s = f.read()
    
    
    if "if __name__ == '__main__':" in s or "if __name__ == \"__main__\":" in s:
        
        return
    
    s += "\n\nif __name__ == '__main__':\n"
    s += "    import sys\n"
    s += "    if len(sys.argv) > 1:\n"
    s += "        video_url = sys.argv[1]\n"
    s += "        start_crawl(video_url)\n"
    s += "    else:\n"
    s += "        print('请提供视频URL作为参数')\n"
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(s)

def _count_csv_rows(csv_path: str) -> int:
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as rf:
        reader = csv.reader(rf)
        rows = list(reader)
    if len(rows) <= 1:
        return 0
    return len(rows) - 1

def _extract_keywords_from_nlp_csv(path: str, topk: int = 12):
    if not os.path.exists(path):
        return []

    df = pd.read_csv(path, encoding="utf-8-sig")
    if "content" not in df.columns:
        return []

    text = " ".join(df["content"].fillna("").astype(str).tolist())
    
    tokens = re.split(r"[\s\r\n\t,.!?;:，。！？；：“”‘’（）()【】\[\]{}<>/\\|@#￥%&*+=~-]+", text)
    tokens = [t.strip() for t in tokens if len(t.strip()) >= 2]
    counter = Counter(tokens)
    return [w for w, _ in counter.most_common(topk)]


@app.route("/api/start-crawl", methods=["POST"])
def start_crawl():
    request_data = request.get_json(force=True) or {}
    video_url = (request_data.get("video_url") or "").strip()
    browser_mode = request_data.get("browser_mode", "headless")

    if not video_url:
        return jsonify({"error": "video_url 不能为空"}), 400

    
    if crawl_data["is_running"]:
        return jsonify({"message": "爬虫正在运行中，请稍后再试"}), 409

    crawl_data.update({
        "is_running": True,
        "progress": 0,
        "total_comments": 0,
        "video_url": video_url,
        "error": None,
        "message": "爬取任务已提交"
    })

    def run_crawler():
        try:
            
            headless_flag = "--headed" if request_data.get('browser_mode') == 'headed' else "--headless"
            
            crawl_data["message"] = "正在启动爬虫脚本..."
            
            result = subprocess.run(
                ["python", CRAWLER_SCRIPT, video_url, headless_flag],
                capture_output=True,
                text=True,
                cwd=DATA_DIR
            )

            if result.returncode != 0:
                crawl_data["error"] = f"爬虫执行失败: {result.stderr or result.stdout}"
                return

            
            total = _count_csv_rows(CRAWLER_OUT_CSV)
            crawl_data["total_comments"] = total
            crawl_data["progress"] = 100
            crawl_data["message"] = f"爬取完成，写入 {total} 条"

        except Exception as e:
            crawl_data["error"] = f"爬虫执行异常: {str(e)}"
        finally:
            crawl_data["is_running"] = False

    threading.Thread(target=run_crawler, daemon=True).start()
    return jsonify({"video_url": video_url, "message": "爬取任务已提交"})


@app.route("/api/crawl-status", methods=["GET"])
def get_crawl_status():
    return jsonify(crawl_data)


@app.route("/api/start-nlp", methods=["POST"])
def start_nlp():
    
    if nlp_data["is_running"]:
        return jsonify({"message": "NLP 正在运行中，请稍后再试"}), 409

    
    if not os.path.exists(CRAWLER_OUT_CSV):
        return jsonify({"error": "未找到 bili_comments.csv，请先完成爬取"}), 400

    nlp_data.update({
        "is_running": True,
        "progress": 0,
        "keywords": [],
        "error": None,
        "message": "NLP 任务已提交"
    })

    def run_nlp():
        try:
            nlp_data["message"] = "正在启动情绪分析..."
            
            
            def update_progress(progress):
                nlp_data["progress"] = progress
                nlp_data["message"] = f"情绪分析进行中 - {progress}%"
            
            
            result_file = comments_nlp.perform_nlp_analysis(
                input_file=CRAWLER_OUT_CSV,
                output_file=NLP_OUT_CSV,
                progress_callback=update_progress
            )

            
            final_status = comments_nlp.get_nlp_status()
            
            if final_status["error"]:
                nlp_data["error"] = final_status["error"]
                return

            nlp_data["progress"] = 100
            nlp_data["message"] = "情绪分析完成，正在提取关键词..."

            
            kws = _extract_keywords_from_nlp_csv(NLP_OUT_CSV, topk=12)
            nlp_data["keywords"] = kws
            nlp_data["message"] = "情绪分析完成"

        except Exception as e:
            nlp_data["error"] = f"情绪分析执行异常: {str(e)}"
        finally:
            nlp_data["is_running"] = False

    threading.Thread(target=run_nlp, daemon=True).start()
    return jsonify({"message": "NLP分析任务已提交"})


@app.route("/api/nlp-status", methods=["GET"])
def get_nlp_status():
    # 从 comments_nlp 获取最新的状态
    if not nlp_data["is_running"] and os.path.exists(NLP_OUT_CSV):
        # 重新加载最新的分析结果
        try:
            df = pd.read_csv(NLP_OUT_CSV, encoding="utf-8-sig")
            
            # 计算情感分布
            sentiment_counts = df["sentiment"].value_counts()
            total = len(df)
            nlp_data["sentiment"] = {
                "pos": float(sentiment_counts.get("Positive", 0)) / total if total > 0 else 0.0,
                "neu": float(sentiment_counts.get("Neutral", 0)) / total if total > 0 else 0.0,
                "neg": float(sentiment_counts.get("Negative", 0)) / total if total > 0 else 0.0
            }
            
            # 生成时间线数据（示例：按评论顺序分段）
            if total > 0:
                segment_size = max(1, total // 6)  # 分成6个时间段
                timeline_data = []
                for i in range(0, total, segment_size):
                    segment = df.iloc[i:i+segment_size]
                    # 这里可以根据实际需求调整时间线的计算方式
                    timeline_data.append(len(segment))
                nlp_data["timeline"] = timeline_data[:6]  # 取前6个时间段
            
            # 提取关键词
            kws = _extract_keywords_from_nlp_csv(NLP_OUT_CSV, topk=12)
            nlp_data["keywords"] = kws
            
        except Exception as e:
            nlp_data["error"] = f"获取NLP状态时出错: {str(e)}"
    
    return jsonify(nlp_data)

if __name__ == "__main__":
    app.run(debug=False, threaded=True)
