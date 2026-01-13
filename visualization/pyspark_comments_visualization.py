import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, to_date, isnan, when, count
from pyspark.sql.types import IntegerType, FloatType, StringType
CONFIG = {
    "csv_path": "file:///usr/local/hadoop/comments_nlp.csv",
    "img_save_dir": "/usr/local/hadoop/comment_visualizations",
    "fig_size": (12, 6),
    "font_size": 11,  
    "colors": { 
        "Positive": "#2E86AB",
        "Negative": "#C73E1D",
        "Neutral": "#F18F01"
    },
    "field_mapping": {
        "text": "content",
        "sentiment": "sentiment",
        "score": "sentiment_score",
        "time": "create_time"
    },
    "date_format": "yyyy-MM-dd HH:mm:ss",
    "valid_sentiments": ["Positive", "Negative", "Neutral"]  
}
# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
def init_spark():
    try:
        spark = SparkSession.builder \
        .appName("NLP_Comment_Visualization_Optimized") \
        .master("local[*]") \
        .config("spark.sql.debug.maxToStringFields", "100") \
        .getOrCreate()
        print(f"Spark初始化成功")
        return spark
    except Exception as e:
        print(f"Spark初始化失败：{str(e)}")
        exit(1)
def load_and_check_data(spark):
    print(f"\n加载本地文件：{CONFIG['csv_path']}")
    local_path = CONFIG["csv_path"].replace("file://", "")
    if not os.path.exists(local_path):
        print(f"本地文件不存在！路径：{local_path}")
        exit(1)
    # 1. 读取原始数据
    df_raw = spark.read.csv(
        path=CONFIG["csv_path"],
        header=True,
        encoding="utf-8",
        inferSchema=False
    )
    raw_count = df_raw.count()
    print(f"原始数据总量：{raw_count}行")
    # 2. 核心字段缺失统计
    field = CONFIG["field_mapping"]
    core_fields = [field["text"], field["sentiment"], field["score"], field["time"]]
    missing_stats = df_raw.select([
        count(when(isnan(f) | col(f).isNull(), f)).alias(f"{f}_缺失数") 
        for f in core_fields
    ]).collect()[0]
    
    print(f"\n核心字段缺失统计：")
    for f in core_fields:
        missing = missing_stats[f"{f}_缺失数"]
        exist = raw_count - missing
        print(f"   {f}：非缺失{exist}行，缺失{missing}行（{missing/raw_count*100:.1f}%）")
    
    # 3. 分步清洗+过滤情感脏数据
    print(f"\n分步清洗数据：")
    df_step1 = df_raw.dropna(subset=core_fields)
    print(f"步骤1-删除缺失行：{df_step1.count()}行")
    
    df_step2 = df_step1.withColumn(
        field["score"],
        col(field["score"]).cast(FloatType())
    ).filter(
        ~isnan(field["score"]) & ~col(field["score"]).isNull()
    )
    print(f"步骤2-评分转换成功：{df_step2.count()}行")
    
    df_step3 = df_step2.filter(col(field["score"]).between(0.0, 1.0))
    print(f"步骤3-过滤0-1分评分：{df_step3.count()}行")
    
    # 关键：过滤异常情感类型（只保留有效类别）
    df_step4 = df_step3.filter(col(field["sentiment"]).isin(CONFIG["valid_sentiments"]))
    print(f"步骤4-过滤情感脏数据：保留{df_step4.count()}行（仅{CONFIG['valid_sentiments']}）")
    # 衍生字段+日期适配
    df_final = df_step4.withColumn(
        "comment_length",
        length(col(field["text"])).cast(IntegerType())
    ).withColumn(
        "comment_date",
        to_date(col(field["time"]), CONFIG["date_format"])
    )
    print(f"步骤5-衍生字段：{df_final.count()}行（无丢失）")
    
    print(f"\n最终清洗后数据：{df_final.count()}行（可用）")
    return df_final.toPandas()
# 情感分布柱状图
def plot_sentiment_dist(df):
    print("\n生成情感分布图表...")
    field = CONFIG["field_mapping"]
    # 按有效情感类型统计
    sentiment_count = df[field["sentiment"]].value_counts().reindex(CONFIG["valid_sentiments"])
    total = sentiment_count.sum()
    # 绘图：固定配色+横向布局
    fig, ax = plt.subplots(figsize=(10, 5))  # 调整宽高比
    bars = ax.barh(
        sentiment_count.index,  # 横向柱状图
        sentiment_count.values,
        color=[CONFIG["colors"][s] for s in sentiment_count.index]
    )
    # 添加数值+占比标签
    for bar, value in zip(bars, sentiment_count.values):
        width = bar.get_width()
        ax.text(
            width + 1000, bar.get_y() + bar.get_height()/2,
            f"{int(value)} ({value/total*100:.1f}%)",
            ha="left", va="center", fontsize=CONFIG["font_size"]
        )
    # 样式优化
    ax.set_title("NLP评论情感分布", fontsize=CONFIG["font_size"]+3, fontweight="bold", pad=20)
    ax.set_xlabel("评论数量", fontsize=CONFIG["font_size"]+1)
    ax.set_ylabel("情感类型", fontsize=CONFIG["font_size"]+1)
    ax.set_xlim(0, total * 1.1)  # 右侧留空放标签
    ax.grid(axis="x", alpha=0.3, linestyle="--")  # 横向网格更易读
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)  # 隐藏上/右边框
    # 保存
    save_path = f"{CONFIG['img_save_dir']}/1_情感分布_优化版.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"情感分布图保存：{save_path}")
# 主函数
def main():
    if not os.path.exists(CONFIG["img_save_dir"]):
        os.makedirs(CONFIG["img_save_dir"])
    abs_save_dir = os.path.abspath(CONFIG["img_save_dir"])
    print(f"图表保存目录：{abs_save_dir}")
    spark = init_spark()
    try:
        df_pandas = load_and_check_data(spark)
        plot_sentiment_dist(df_pandas)
        print(f"\n 所有图表生成完成！")
        print(f"图表位置：{abs_save_dir}")
    except Exception as e:
        print(f"\n执行失败：{str(e)}")
    finally:
        spark.stop()
        print("\nSpark会话已关闭")
if __name__ == "__main__":
    main()