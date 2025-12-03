"""
TF-IDF关键词提取 & 争议性分析 - 运行入口

==========================================================================
大数据分析方法:
==========================================================================

1. TF-IDF (Term Frequency - Inverse Document Frequency)
   - TF: 词在文档中的频率
   - IDF: log(总文档数 / 包含该词的文档数)  
   - 用于识别各subreddit的特色关键词

2. MapReduce 模式
   - Map: 对每条评论分词，输出(word, 1)
   - Reduce: 按组聚合，统计词频
   
3. 向量化操作
   - 使用pandas向量操作替代循环
   - 提升10-100倍性能

4. 分层抽样
   - 对比分析时平衡样本

==========================================================================

使用方法:
    python -m HotCommentLens.run_analysis
    python -m HotCommentLens.run_analysis --sample 100000
"""

import os
import sys
import time
import argparse
import pandas as pd
from datetime import datetime

from .config import Config
from .data_loader import RedditDataLoader
from .tfidf_controversy import TFIDFAnalyzer, ControversyAnalyzer, AnalysisVisualizer


def main():
    parser = argparse.ArgumentParser(
        description='TF-IDF关键词提取 & 争议性分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
大数据分析方法:
  1. TF-IDF: 关键词提取，识别社区特色词汇
  2. MapReduce: 大规模词频统计
  3. 向量化: pandas高效计算
  4. 分层抽样: 平衡对比分析

示例:
  python -m HotCommentLens.run_analysis
  python -m HotCommentLens.run_analysis --sample 100000
        """
    )
    parser.add_argument('--sample', type=int, default=None,
                       help='抽样数量 (默认: 全部数据)')
    parser.add_argument('--data', type=str, default=None,
                       help='数据文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 输出目录
    output_dir = args.output or os.path.join(Config.OUTPUT_DIR, "tfidf_controversy")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("🔬 Reddit评论分析: TF-IDF关键词 + 争议性分析")
    print("="*70)
    
    print("""
📚 使用的大数据方法:
   1. TF-IDF (Term Frequency - Inverse Document Frequency)
      → 识别各subreddit特色关键词
   2. MapReduce模式
      → Map: 文档分词  Reduce: 词频聚合
   3. 向量化操作
      → pandas向量计算，替代Python循环
   4. 分层抽样
      → 平衡争议性/非争议性样本对比
""")
    
    # ==================== 加载数据 ====================
    print("="*50)
    print("📂 [数据加载]")
    print("="*50)
    
    start_time = time.time()
    
    loader = RedditDataLoader()
    if args.data:
        df = pd.read_csv(args.data)
    else:
        df = loader.load(num_comments=args.sample)
    
    # 适配列名
    if 'text' in df.columns and 'body' not in df.columns:
        df['body'] = df['text']
    
    load_time = time.time() - start_time
    print(f"\n✅ 加载完成: {len(df):,} 条评论 ({load_time:.2f}秒)")
    
    # 数据验证
    required_cols = ['subreddit', 'body', 'controversiality', 'score']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ 缺少必要列: {missing}")
        sys.exit(1)
    
    # 过滤
    original_len = len(df)
    df = df.dropna(subset=['body', 'subreddit'])
    df = df[df['body'].str.len() >= 10]
    print(f"📊 有效评论: {len(df):,} ({len(df)/original_len*100:.1f}%)")
    print(f"📁 Subreddits: {df['subreddit'].nunique()}")
    
    # ==================== 第一部分: TF-IDF ====================
    print("\n" + "="*50)
    print("📌 第一部分: TF-IDF关键词提取")
    print("="*50)
    
    tfidf_start = time.time()
    tfidf_analyzer = TFIDFAnalyzer()
    tfidf_results = tfidf_analyzer.compute_tfidf(df, text_column='body', group_column='subreddit')
    tfidf_time = time.time() - tfidf_start
    
    print(f"\n🔑 各Subreddit TF-IDF关键词:")
    for sub in list(tfidf_results.keys())[:6]:
        keywords = tfidf_results[sub][:5]
        kw_str = ', '.join([w for w, _ in keywords])
        print(f"   r/{sub}: {kw_str}")
    
    # ==================== 第二部分: 争议性分析 ====================
    print("\n" + "="*50)
    print("📌 第二部分: 争议性分析")
    print("="*50)
    
    controversy_start = time.time()
    controversy_analyzer = ControversyAnalyzer()
    stats = controversy_analyzer.analyze(df, text_column='body')
    controversy_time = time.time() - controversy_start
    
    # ==================== 可视化 ====================
    print("\n" + "="*50)
    print("📊 生成可视化")
    print("="*50)
    
    viz = AnalysisVisualizer(output_dir)
    viz.plot_tfidf_keywords(tfidf_results)
    viz.plot_controversy_stats(stats)
    viz.plot_keyword_comparison(stats.controversial_keywords, stats.non_controversial_keywords)
    
    # ==================== 导出结果 ====================
    print("\n" + "="*50)
    print("💾 导出结果")
    print("="*50)
    
    # TF-IDF结果
    tfidf_rows = []
    for sub, keywords in tfidf_results.items():
        for rank, (word, score) in enumerate(keywords, 1):
            tfidf_rows.append({
                'subreddit': sub,
                'rank': rank,
                'keyword': word,
                'tfidf_score': round(score, 6)
            })
    tfidf_df = pd.DataFrame(tfidf_rows)
    tfidf_df.to_csv(os.path.join(output_dir, 'tfidf_keywords.csv'),
                    index=False, encoding='utf-8-sig')
    try:
        tfidf_df.to_excel(os.path.join(output_dir, 'tfidf_keywords.xlsx'), index=False)
        print(f"   ✅ tfidf_keywords.csv/xlsx")
    except ImportError:
        print(f"   ✅ tfidf_keywords.csv")
    
    # 争议性统计
    stats_data = {
        'Metric': ['Total Comments', 'Controversial', 'Non-controversial',
                   'Controversy Rate (%)', 'Avg Length (Controversial)',
                   'Avg Length (Non-controversial)', 'Avg Score (Controversial)',
                   'Avg Score (Non-controversial)'],
        'Value': [stats.total_comments, stats.controversial_count,
                  stats.non_controversial_count, f"{stats.controversy_rate:.2f}",
                  f"{stats.avg_length_controversial:.1f}",
                  f"{stats.avg_length_non_controversial:.1f}",
                  f"{stats.avg_score_controversial:.2f}",
                  f"{stats.avg_score_non_controversial:.2f}"]
    }
    pd.DataFrame(stats_data).to_csv(
        os.path.join(output_dir, 'controversy_stats.csv'),
        index=False, encoding='utf-8-sig'
    )
    print(f"   ✅ controversy_stats.csv")
    
    # Subreddit争议率
    sub_rates = pd.DataFrame([
        {'subreddit': sub, 'controversy_rate': rate}
        for sub, rate in stats.subreddit_controversy_rates.items()
    ]).sort_values('controversy_rate', ascending=False)
    sub_rates.to_csv(os.path.join(output_dir, 'subreddit_controversy_rates.csv'),
                     index=False, encoding='utf-8-sig')
    print(f"   ✅ subreddit_controversy_rates.csv")
    
    # 关键词对比
    kw_rows = []
    for word, score in stats.controversial_keywords[:30]:
        kw_rows.append({'type': 'controversial', 'keyword': word, 'score': score})
    for word, score in stats.non_controversial_keywords[:30]:
        kw_rows.append({'type': 'non_controversial', 'keyword': word, 'score': score})
    pd.DataFrame(kw_rows).to_csv(
        os.path.join(output_dir, 'keyword_comparison.csv'),
        index=False, encoding='utf-8-sig'
    )
    print(f"   ✅ keyword_comparison.csv")
    
    # ==================== 完成 ====================
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ 分析完成!")
    print("="*70)
    
    print(f"""
📁 输出目录: {output_dir}
   - visualizations/
     - tfidf_keywords.png
     - controversy_analysis.png  
     - keyword_comparison.png
   - tfidf_keywords.csv/xlsx
   - controversy_stats.csv
   - subreddit_controversy_rates.csv
   - keyword_comparison.csv

⏱️  耗时统计:
   数据加载: {load_time:.2f}秒
   TF-IDF分析: {tfidf_time:.2f}秒
   争议性分析: {controversy_time:.2f}秒
   总计: {total_time:.2f}秒

📚 使用的大数据方法:
   ✓ TF-IDF - 关键词提取
   ✓ MapReduce - 词频统计
   ✓ 向量化操作 - 高效计算
   ✓ 分层抽样 - 平衡对比
""")


if __name__ == '__main__':
    main()
