"""
TF-IDF关键词提取 & 争议性分析模块

==========================================================================
大数据分析方法说明:
==========================================================================

1. TF-IDF (Term Frequency - Inverse Document Frequency)
   -------------------------------------------------------
   - TF (词频): 词在文档中出现的次数 / 文档总词数
   - IDF (逆文档频率): log(总文档数 / 包含该词的文档数)
   - TF-IDF = TF × IDF
   - 作用: 识别在特定社区中重要但全局不常见的关键词
   - 优点: 自动降低常见词权重，突出特色词汇

2. MapReduce 模式
   -------------------------------------------------------
   - Map阶段: 对每条评论进行分词，提取(word, 1)键值对
   - Reduce阶段: 聚合相同词的计数，得到词频统计
   - 应用场景: 大规模文本词频统计、分组聚合
   - 本项目应用: 按subreddit分组统计词频

3. 批量处理 (Batch Processing)
   -------------------------------------------------------
   - 将100万条数据分成多个小批次处理
   - 每批处理完后释放内存
   - 避免一次性加载导致内存溢出

4. 向量化操作 (Vectorization)
   -------------------------------------------------------
   - 使用pandas/numpy替代Python循环
   - 利用底层C优化，提升10-100倍性能
   - 例: df['length'] = df['text'].str.len()

5. 分层抽样 (Stratified Sampling)
   -------------------------------------------------------
   - 从争议性/非争议性评论中按比例抽样
   - 保证对比分析的公平性和代表性
==========================================================================
"""

import os
import re
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

# 配置
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .config import Config
except ImportError:
    from config import Config


@dataclass
class ControversyStats:
    """争议性统计结果"""
    total_comments: int = 0
    controversial_count: int = 0
    non_controversial_count: int = 0
    controversy_rate: float = 0.0
    
    # 特征对比
    avg_length_controversial: float = 0.0
    avg_length_non_controversial: float = 0.0
    avg_score_controversial: float = 0.0
    avg_score_non_controversial: float = 0.0
    
    # 关键词
    controversial_keywords: List[Tuple[str, float]] = field(default_factory=list)
    non_controversial_keywords: List[Tuple[str, float]] = field(default_factory=list)
    
    # 按subreddit统计
    subreddit_controversy_rates: Dict[str, float] = field(default_factory=dict)


class TFIDFAnalyzer:
    """
    TF-IDF关键词提取器
    
    大数据方法:
    ===========
    1. MapReduce模式进行词频统计
    2. 批量处理避免内存溢出
    3. 向量化操作加速计算
    """
    
    def __init__(self):
        self.stopwords = self._get_stopwords()
    
    def _get_stopwords(self) -> set:
        """获取停用词表"""
        base_stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
            'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
            'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
            'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
            'with', 'about', 'against', 'between', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
            "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
        }
        
        extra_stopwords = {
            'would', 'could', 'get', 'got', 'like', 'know', 'think',
            'really', 'even', 'well', 'also', 'still', 'way', 'much',
            'thing', 'things', 'something', 'anything', 'nothing',
            'people', 'person', 'one', 'two', 'first', 'new', 'good',
            'make', 'made', 'see', 'want', 'say', 'said', 'going',
            'take', 'come', 'came', 'look', 'use', 'used', 'time',
            'yeah', 'yes', 'okay', 'actually', 'probably', 'maybe',
            'right', 'need', 'mean', 'sure', 'lot', 'back',
            'thats', 'dont', 'doesnt', 'didnt', 'cant', 'wont', 'isnt',
            'im', 'ive', 'youre', 'hes', 'shes', 'theyre', 'wasnt',
            'deleted', 'removed', 'comment', 'edit', 'reddit', 'sub',
            'http', 'https', 'www', 'com', 'org', 'amp', 'any',
        }
        
        return base_stopwords | extra_stopwords
    
    def compute_tfidf(
        self, 
        df: pd.DataFrame,
        text_column: str = 'body',
        group_column: str = 'subreddit',
        top_n: int = 20
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        计算各组的TF-IDF关键词
        
        大数据方法:
        -----------
        1. MapReduce: 
           - Map: 每个文档分词 -> (word, doc_id)
           - Reduce: 按组聚合词频
        
        2. 向量化: 使用pandas groupby进行高效分组
        
        Args:
            df: 数据DataFrame
            text_column: 文本列名
            group_column: 分组列名
            top_n: 每组返回的关键词数量
            
        Returns:
            {group: [(word, tfidf_score), ...]}
        """
        print("\n" + "="*60)
        print("📊 TF-IDF 关键词提取")
        print("="*60)
        print("\n🔧 大数据方法: MapReduce + 向量化操作")
        
        groups = df[group_column].unique()
        total_docs = len(df)
        print(f"\n📁 分组数: {len(groups)}")
        print(f"📄 总文档数: {total_docs:,}")
        
        # ============ Map阶段: 统计每组词频 ============
        print("\n🗺️  [Map阶段] 统计各组词频...")
        group_word_counts = {}  # {group: Counter}
        group_doc_counts = {}   # {group: doc_count}
        
        for group in groups:
            group_df = df[df[group_column] == group]
            group_doc_counts[group] = len(group_df)
            
            # 批量统计词频
            word_counter = Counter()
            for text in group_df[text_column]:
                if isinstance(text, str):
                    words = re.findall(r'\b[a-z]{3,15}\b', text.lower())
                    words = [w for w in words if w not in self.stopwords]
                    # 使用set去重，统计文档频率(DF)
                    word_counter.update(set(words))
            
            group_word_counts[group] = word_counter
        
        # ============ 计算全局文档频率(IDF) ============
        print("📉 [计算IDF] 统计全局文档频率...")
        global_doc_freq = Counter()
        for counter in group_word_counts.values():
            global_doc_freq.update(counter.keys())
        
        # ============ Reduce阶段: 计算TF-IDF ============
        print("🔢 [Reduce阶段] 计算TF-IDF得分...")
        tfidf_results = {}
        
        for group in groups:
            word_counts = group_word_counts[group]
            group_total = sum(word_counts.values())
            
            if group_total == 0:
                tfidf_results[group] = []
                continue
            
            word_scores = []
            for word, count in word_counts.items():
                # TF: 词频 / 总词数
                tf = count / group_total
                
                # IDF: log(总文档数 / 包含该词的文档数)
                df_count = global_doc_freq.get(word, 1)
                idf = np.log(total_docs / df_count)
                
                # TF-IDF
                tfidf = tf * idf
                
                if count >= 5:  # 最小频率阈值
                    word_scores.append((word, tfidf, count))
            
            # 排序取top_n
            word_scores.sort(key=lambda x: x[1], reverse=True)
            tfidf_results[group] = [(w, score) for w, score, _ in word_scores[:top_n]]
        
        print("✅ TF-IDF计算完成!")
        return tfidf_results


class ControversyAnalyzer:
    """
    争议性分析器
    
    大数据方法:
    ===========
    1. 向量化操作: pandas布尔索引、groupby聚合
    2. MapReduce: 分组统计各指标
    3. 分层抽样: 对比分析时保证样本平衡
    """
    
    def __init__(self):
        self.stopwords = TFIDFAnalyzer()._get_stopwords()
    
    def analyze(self, df: pd.DataFrame, text_column: str = 'body') -> ControversyStats:
        """
        执行争议性分析
        
        大数据方法:
        -----------
        - 向量化: 使用pandas向量操作替代循环
        - MapReduce: groupby = Map, agg = Reduce
        """
        print("\n" + "="*60)
        print("🔥 争议性分析 (Controversy Analysis)")
        print("="*60)
        print("\n🔧 大数据方法: 向量化操作 + MapReduce聚合")
        
        stats = ControversyStats()
        
        # ============ 1. 基础统计 (向量化) ============
        print("\n📊 [1/4] 基础统计 (向量化操作)...")
        stats.total_comments = len(df)
        
        # 布尔索引 - 向量化操作
        controversial_mask = df['controversiality'] == 1
        stats.controversial_count = controversial_mask.sum()
        stats.non_controversial_count = stats.total_comments - stats.controversial_count
        stats.controversy_rate = stats.controversial_count / stats.total_comments * 100
        
        print(f"   总评论数: {stats.total_comments:,}")
        print(f"   争议性评论: {stats.controversial_count:,} ({stats.controversy_rate:.2f}%)")
        print(f"   非争议性评论: {stats.non_controversial_count:,}")
        
        # ============ 2. 特征对比 (向量化) ============
        print("\n📏 [2/4] 特征对比 (向量化计算)...")
        
        # 向量化计算评论长度
        df = df.copy()
        df['comment_length'] = df[text_column].str.len()
        
        controversial_df = df[controversial_mask]
        non_controversial_df = df[~controversial_mask]
        
        stats.avg_length_controversial = controversial_df['comment_length'].mean()
        stats.avg_length_non_controversial = non_controversial_df['comment_length'].mean()
        stats.avg_score_controversial = controversial_df['score'].mean()
        stats.avg_score_non_controversial = non_controversial_df['score'].mean()
        
        print(f"   争议评论平均长度: {stats.avg_length_controversial:.1f} 字符")
        print(f"   非争议评论平均长度: {stats.avg_length_non_controversial:.1f} 字符")
        print(f"   争议评论平均得分: {stats.avg_score_controversial:.2f}")
        print(f"   非争议评论平均得分: {stats.avg_score_non_controversial:.2f}")
        
        # ============ 3. 按Subreddit统计 (MapReduce) ============
        print("\n📁 [3/4] 按Subreddit统计 (MapReduce: groupby+agg)...")
        
        # groupby = Map, agg = Reduce
        subreddit_stats = df.groupby('subreddit').agg({
            'controversiality': ['sum', 'count']
        })
        subreddit_stats.columns = ['controversial', 'total']
        subreddit_stats['rate'] = subreddit_stats['controversial'] / subreddit_stats['total'] * 100
        subreddit_stats = subreddit_stats.sort_values('rate', ascending=False)
        
        stats.subreddit_controversy_rates = subreddit_stats['rate'].to_dict()
        
        print(f"   最具争议性的Subreddit Top 5:")
        for i, (sub, row) in enumerate(subreddit_stats.head(5).iterrows()):
            print(f"   {i+1}. r/{sub}: {row['rate']:.2f}% ({int(row['controversial'])}/{int(row['total'])})")
        
        # ============ 4. 提取差异关键词 (分层抽样 + MapReduce) ============
        print("\n🔑 [4/4] 差异关键词 (分层抽样 + MapReduce)...")
        stats.controversial_keywords, stats.non_controversial_keywords = \
            self._extract_differential_keywords(
                controversial_df, non_controversial_df, text_column
            )
        
        print(f"   争议性高频词: {', '.join([w for w, _ in stats.controversial_keywords[:8]])}")
        print(f"   非争议性高频词: {', '.join([w for w, _ in stats.non_controversial_keywords[:8]])}")
        
        return stats
    
    def _extract_differential_keywords(
        self,
        controversial_df: pd.DataFrame,
        non_controversial_df: pd.DataFrame,
        text_column: str,
        sample_size: int = 50000
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        提取差异性关键词
        
        大数据方法:
        -----------
        - 分层抽样: 平衡两类样本
        - MapReduce: 批量词频统计
        """
        # 分层抽样
        if len(controversial_df) > sample_size:
            controversial_sample = controversial_df.sample(n=sample_size, random_state=42)
        else:
            controversial_sample = controversial_df
            
        if len(non_controversial_df) > sample_size:
            non_controversial_sample = non_controversial_df.sample(n=sample_size, random_state=42)
        else:
            non_controversial_sample = non_controversial_df
        
        # MapReduce词频统计
        controversial_words = self._count_words(controversial_sample[text_column])
        non_controversial_words = self._count_words(non_controversial_sample[text_column])
        
        # 计算差异性得分
        controversial_kw = self._compute_differential_score(controversial_words, non_controversial_words)
        non_controversial_kw = self._compute_differential_score(non_controversial_words, controversial_words)
        
        return controversial_kw, non_controversial_kw
    
    def _count_words(self, texts: pd.Series) -> Counter:
        """批量统计词频 (MapReduce)"""
        counter = Counter()
        for text in texts:
            if isinstance(text, str):
                words = re.findall(r'\b[a-z]{3,15}\b', text.lower())
                words = [w for w in words if w not in self.stopwords]
                counter.update(words)
        return counter
    
    def _compute_differential_score(
        self,
        target_counter: Counter,
        background_counter: Counter,
        top_n: int = 50
    ) -> List[Tuple[str, float]]:
        """计算差异性得分"""
        target_total = sum(target_counter.values())
        background_total = sum(background_counter.values())
        
        if target_total == 0 or background_total == 0:
            return []
        
        scores = []
        for word, count in target_counter.most_common(300):
            target_freq = count / target_total
            background_freq = (background_counter.get(word, 0) + 1) / (background_total + 1)
            diff_score = target_freq / background_freq
            
            if count >= 10:
                scores.append((word, diff_score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


class AnalysisVisualizer:
    """分析结果可视化"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_tfidf_keywords(
        self,
        tfidf_results: Dict[str, List[Tuple[str, float]]],
        top_n_groups: int = 6,
        top_n_keywords: int = 10
    ) -> str:
        """绘制TF-IDF关键词图"""
        # 选取关键词最多的组
        groups = sorted(tfidf_results.keys(),
                       key=lambda x: len(tfidf_results[x]),
                       reverse=True)[:top_n_groups]
        
        n_cols = 2
        n_rows = (len(groups) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
        axes = axes.flatten()
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
        
        for idx, group in enumerate(groups):
            ax = axes[idx]
            keywords = tfidf_results[group][:top_n_keywords]
            
            if not keywords:
                ax.text(0.5, 0.5, 'No keywords', ha='center', va='center')
                ax.set_title(f'r/{group}')
                continue
            
            words = [w for w, _ in keywords]
            scores = [s for _, s in keywords]
            
            y_pos = np.arange(len(words))
            ax.barh(y_pos, scores, color=colors[idx], alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words)
            ax.set_xlabel('TF-IDF Score')
            ax.set_title(f'r/{group}', fontsize=11, fontweight='bold')
            ax.invert_yaxis()
        
        for idx in range(len(groups), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('TF-IDF Keywords by Subreddit', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        filepath = os.path.join(self.viz_dir, 'tfidf_keywords.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   保存: {filepath}")
        return filepath
    
    def plot_controversy_stats(self, stats: ControversyStats) -> str:
        """绘制争议性统计图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 饼图
        ax1 = axes[0, 0]
        sizes = [stats.controversial_count, stats.non_controversial_count]
        labels = [f'Controversial\n({stats.controversial_count:,})',
                  f'Non-controversial\n({stats.non_controversial_count:,})']
        colors = ['#e74c3c', '#3498db']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, explode=(0.05, 0))
        ax1.set_title('Comment Distribution', fontsize=12, fontweight='bold')
        
        # 2. 长度对比
        ax2 = axes[0, 1]
        categories = ['Controversial', 'Non-controversial']
        lengths = [stats.avg_length_controversial, stats.avg_length_non_controversial]
        bars = ax2.bar(categories, lengths, color=['#e74c3c', '#3498db'])
        ax2.set_ylabel('Average Length (chars)')
        ax2.set_title('Comment Length Comparison', fontsize=12, fontweight='bold')
        for bar, val in zip(bars, lengths):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    f'{val:.0f}', ha='center', fontsize=10)
        
        # 3. 得分对比
        ax3 = axes[1, 0]
        scores = [stats.avg_score_controversial, stats.avg_score_non_controversial]
        bars = ax3.bar(categories, scores, color=['#e74c3c', '#3498db'])
        ax3.set_ylabel('Average Score')
        ax3.set_title('Comment Score Comparison', fontsize=12, fontweight='bold')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        for bar, val in zip(bars, scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.2f}', ha='center', fontsize=10)
        
        # 4. Top争议subreddit
        ax4 = axes[1, 1]
        top_subs = dict(sorted(stats.subreddit_controversy_rates.items(),
                               key=lambda x: x[1], reverse=True)[:10])
        subs = list(top_subs.keys())
        rates = list(top_subs.values())
        
        y_pos = np.arange(len(subs))
        ax4.barh(y_pos, rates, color='#e74c3c', alpha=0.8)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([f'r/{s}' for s in subs])
        ax4.set_xlabel('Controversy Rate (%)')
        ax4.set_title('Top 10 Controversial Subreddits', fontsize=12, fontweight='bold')
        ax4.invert_yaxis()
        
        for bar, val in zip(ax4.patches, rates):
            ax4.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.viz_dir, 'controversy_analysis.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   保存: {filepath}")
        return filepath
    
    def plot_keyword_comparison(
        self,
        controversial_kw: List[Tuple[str, float]],
        non_controversial_kw: List[Tuple[str, float]],
        top_n: int = 15
    ) -> str:
        """绘制关键词对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        
        # 争议性关键词
        ax1 = axes[0]
        words1 = [w for w, _ in controversial_kw[:top_n]]
        scores1 = [s for _, s in controversial_kw[:top_n]]
        y_pos = np.arange(len(words1))
        ax1.barh(y_pos, scores1, color='#e74c3c', alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(words1)
        ax1.set_xlabel('Differential Score')
        ax1.set_title('Controversial Keywords', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        
        # 非争议性关键词
        ax2 = axes[1]
        words2 = [w for w, _ in non_controversial_kw[:top_n]]
        scores2 = [s for _, s in non_controversial_kw[:top_n]]
        y_pos = np.arange(len(words2))
        ax2.barh(y_pos, scores2, color='#3498db', alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(words2)
        ax2.set_xlabel('Differential Score')
        ax2.set_title('Non-controversial Keywords', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        
        filepath = os.path.join(self.viz_dir, 'keyword_comparison.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   保存: {filepath}")
        return filepath
