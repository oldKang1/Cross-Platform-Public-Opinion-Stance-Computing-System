import pandas as pd
import time
from transformers import pipeline
import numpy as np
import os
import json
import requests
from pathlib import Path

from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

thr = 0.94

NLP_STATUS = {
    "is_running": False,
    "progress": 0,
    "total_comments": 0,
    "current_task": "",
    "error": None,
    "sentiment": {"pos": 0.0, "neu": 0.0, "neg": 0.0},
    "timeline": []
}

def to_neutral(label, score):
    return "Neutral" if float(score) < thr else label

def perform_nlp_analysis(input_file="bili_comments.csv", output_file="bili_comments_nlp.csv", progress_callback=None):
    global NLP_STATUS
    NLP_STATUS["is_running"] = True
    NLP_STATUS["progress"] = 0
    NLP_STATUS["current_task"] = f"正在分析 {input_file} 中的评论"
    NLP_STATUS["error"] = None
    
    try:
        print("开始加载评论数据...")
        df = pd.read_csv(input_file, encoding="utf-8")
        
        text_col = "content"
        
        df[text_col] = df[text_col].replace(np.nan, "").astype(str).str.strip()
        

        df = df[df[text_col].str.len() > 0]
        
        total_comments = len(df)
        NLP_STATUS["total_comments"] = total_comments
        
        if total_comments == 0:
            print("没有找到需要分析的评论")
            NLP_STATUS["progress"] = 100
            NLP_STATUS["is_running"] = False
            return output_file
        
        print(f"正在加载模型，需要分析 {total_comments} 条评论")
        device_id = 0
        pipe = pipeline(
            "sentiment-analysis",
            model="IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment",
            device=device_id
        )
        
        print("开始情绪分析...")
        
        ds = Dataset.from_dict({"text": df[text_col].tolist()})
        
        batch_size = 8
        
        labels, scores = [], []
        

        for i, out in enumerate(pipe(
            KeyDataset(ds, "text"),
            batch_size=batch_size,
            truncation=True,
            max_length=512
        )):
            if isinstance(out, list):
                for item in out:
                    labels.append(item["label"])
                    scores.append(float(item["score"]))
            else:
                labels.append(out["label"])
                scores.append(float(out["score"]))
            

            current_progress = int(((i + 1) / total_comments) * 100)
            NLP_STATUS["progress"] = min(current_progress, 99)
            

            if progress_callback:
                progress_callback(NLP_STATUS["progress"])
            

            if (i + 1) % 500 == 0 or i == total_comments - 1:
                print(f"情绪分析进度: {i + 1}/{total_comments} ({NLP_STATUS['progress']}%)")
        
        df["sentiment"] = labels
        df["sentiment_score"] = scores
        df["sentiment"] = [to_neutral(l, s) for l, s in zip(df["sentiment"], df["sentiment_score"])]
        
        # 计算情感分布
        sentiment_counts = df["sentiment"].value_counts()
        total = len(df)
        NLP_STATUS["sentiment"] = {
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
                # 例如：计算每个时间段的情感得分平均值或评论数量
                timeline_data.append(len(segment))
            NLP_STATUS["timeline"] = timeline_data[:6]  # 取前6个时间段
        
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        
        NLP_STATUS["progress"] = 100
        NLP_STATUS["is_running"] = False
        print(f"情绪分析完成 -> {output_file}")
        print(df["sentiment"].value_counts())
        
        return output_file
        
    except Exception as e:
        NLP_STATUS["error"] = str(e)
        NLP_STATUS["is_running"] = False
        NLP_STATUS["progress"] = 0
        print(f"情绪分析出错: {str(e)}")
        raise e

def get_nlp_status():
    return NLP_STATUS.copy()

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "bili_comments.csv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "bili_comments_nlp.csv"
    
    def print_progress(progress):
        print(f"情绪分析进行中 - {progress}%")
    
    perform_nlp_analysis(input_file, output_file, print_progress)
