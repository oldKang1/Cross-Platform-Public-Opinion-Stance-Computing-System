import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, to_date, isnan, when, count, avg
from pyspark.sql.types import IntegerType, FloatType, StringType

# --------------------------
# 配置参数
# --------------------------
CONFIG = {
    "csv_path": "file:///usr/local/hadoop/comments_nlp_utf8.csv",
    "img_save_dir": "./comment_visualizations/地域分析",
    "fig_size": (20, 12),
    "colors": {
        "Positive": "#2E86AB",
        "Negative": "#C73E1D",
        "Neutral": "#F18F01"
    },
    "field_mapping": {
        "text": "content",
        "sentiment": "sentiment",
        "score": "sentiment_score",
        "time": "create_time",
        "region": "ip_location"
    },
    "valid_sentiments": ["Positive", "Negative", "Neutral"],
    "min_comment_threshold": 50,
    "top_regions": 15
}
BOLD_FONT_PATH = "/home/hadoop/matplotlib_fonts/NotoSansCJKSC-Bold.ttf"
if not os.path.exists(BOLD_FONT_PATH):
    print(f"字体文件不存在：{BOLD_FONT_PATH}")
    exit(1)
bold_font = fm.FontProperties(
    fname=BOLD_FONT_PATH,
    size=16
)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 16
def init_spark():
    """初始化SparkSession"""
    try:
        spark = (
            SparkSession.builder
            .appName("NLP_Region_Analysis_Final")
            .master("local[*]")
            .config("spark.sql.debug.maxToStringFields", "100")
            .getOrCreate()
        )
        print(f"✅ Spark初始化成功")
        return spark
    except Exception as e:
        print(f"Spark初始化失败：{str(e)}")
        exit(1)

def load_region_data(spark):
    """加载并预处理地域数据"""
    print(f"\n加载数据：{CONFIG['csv_path']}")
    local_path = CONFIG["csv_path"].replace("file://", "")
    if not os.path.exists(local_path):
        print(f" 数据文件不存在：{local_path}")
        exit(1)
    field = CONFIG["field_mapping"]
    df_raw = spark.read.csv(
        path=CONFIG["csv_path"],
        header=True,
        encoding="utf-8",
        inferSchema=False
    )
    core_fields = [field["text"], field["sentiment"], field["score"], field["region"]]
    missing_stats = df_raw.select([
        count(when(isnan(f) | col(f).isNull(), f)).alias(f"{f}_缺失数") 
        for f in core_fields
    ]).collect()[0]
    print(f"\n核心字段缺失统计：")
    for f in core_fields:
        missing = missing_stats[f"{f}_缺失数"]
        exist = df_raw.count() - missing
        print(f"   {f}：非缺失{exist}行，缺失{missing}行（{missing/df_raw.count()*100:.1f}%）")
    # 数据清洗
    df_clean = (
        df_raw
        .dropna(subset=core_fields)
        .withColumn(field["score"], col(field["score"]).cast(FloatType()))
        .filter(col(field["score"]).between(0.0, 1.0))
        .filter(col(field["sentiment"]).isin(CONFIG["valid_sentiments"]))
        .filter(col(field["region"]) != "未知")
    )
    # 过滤小样本地区
    region_comment_count = (
        df_clean
        .groupBy(field["region"])
        .agg(count(field["text"]).alias("comment_total"))
        .filter(col("comment_total") >= CONFIG["min_comment_threshold"])
    )
    
    df_region = df_clean.join(region_comment_count, on=field["region"], how="inner")
    print(f"\n数据预处理完成：")
    print(f"   - 有效地域：{region_comment_count.count()}个")
    print(f"   - 可用数据：{df_region.count()}行")
    return df_region.toPandas()

# --------------------------
# 图表：各地区评论情感分布堆叠图
# --------------------------
def plot_region_sentiment_stack(df):
    print(f"\n生成【各地区情感分布堆叠图】...")
    field = CONFIG["field_mapping"]
    # 数据准备
    region_sentiment = (
        df.groupby([field["region"], field["sentiment"]])
        .agg({field["text"]: "count"})
        .rename(columns={field["text"]: "comment_count"})
        .reset_index()
    )
    region_sentiment_pivot = (
        region_sentiment.pivot(
            index=field["region"],
            columns=field["sentiment"],
            values="comment_count"
        )
        .fillna(0)
    )
    region_sentiment_pivot["total"] = region_sentiment_pivot.sum(axis=1)
    region_sentiment_sorted = (
        region_sentiment_pivot
        .sort_values("total", ascending=False)
        .head(CONFIG["top_regions"])
    )
    # 绘图
    fig, ax = plt.subplots(figsize=CONFIG["fig_size"])
    sentiment_order = ["Positive", "Neutral", "Negative"]
    bottom = np.zeros(len(region_sentiment_sorted))
    for sentiment in sentiment_order:
        if sentiment in region_sentiment_sorted.columns:
            ax.bar(
                region_sentiment_sorted.index,
                region_sentiment_sorted[sentiment],
                bottom=bottom,
                label=sentiment,
                color=CONFIG["colors"][sentiment],
                alpha=0.8
            )
            bottom += region_sentiment_sorted[sentiment]
    # 标题
    ax.set_title(
        f"各地区评论情感分布（前{CONFIG['top_regions']}个热门地区）",
        fontproperties=bold_font,
        fontsize=22,
        pad=30
    )
    # X轴标签（加粗）
    ax.set_xlabel("地域", fontproperties=bold_font, fontsize=20)
    # Y轴标签（加粗）
    ax.set_ylabel("评论数量", fontproperties=bold_font, fontsize=20)
    # 地域刻度（加粗）
    ax.set_xticks(range(len(region_sentiment_sorted.index)))
    ax.set_xticklabels(
        region_sentiment_sorted.index,
        rotation=30,
        ha="right",
        fontproperties=bold_font,
        fontsize=18
    )
    # Y轴刻度（加粗）
    ax.set_yticklabels(
        ax.get_yticks(),
        fontproperties=bold_font,
        fontsize=18
    )
    # 图例（加粗）
    legend = ax.legend(
        title="情感类型",
        prop=bold_font,
        fontsize=18,
        loc="upper right",
        handlelength=3,
        handleheight=2,
        borderpad=2,
        labelspacing=2,
        handletextpad=2
    )
    legend.get_title().set_fontproperties(bold_font)
    legend.get_title().set_fontsize(20)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=1.5)
    # 保存
    save_path = f"{CONFIG['img_save_dir']}/1_各地区情感分布堆叠图.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"堆叠图保存：{save_path}")

# --------------------------
# 主函数
# --------------------------
def main():
    # 创建图表保存目录
    if not os.path.exists(CONFIG["img_save_dir"]):
        os.makedirs(CONFIG["img_save_dir"])
    abs_save_dir = os.path.abspath(CONFIG["img_save_dir"])
    print(f"图表保存目录：{abs_save_dir}")
    
    # 初始化Spark并执行分析
    spark = init_spark()
    try:
        df_region = load_region_data(spark)
        plot_region_sentiment_stack(df_region)  # 各地区情感分布堆叠图
        
        print(f"\n图表生成完成！")
        print(f" 图表已保存至：{abs_save_dir}")
    except Exception as e:
        print(f"\n执行失败：{str(e)}")
    finally:
        spark.stop()
        print("\n Spark会话已关闭")

if __name__ == "__main__":
    main()