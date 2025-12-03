"""
Reddit数据加载与预处理模块

支持加载Kaggle Reddit评论数据集:
https://www.kaggle.com/datasets/smagnan/1-million-reddit-comments-from-40-subreddits
"""

import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Generator
from collections import Counter
from datetime import datetime

from .config import Config


class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or Config.PREPROCESSING
        self.stopwords = Config.get_all_stopwords()
        
        # 编译正则表达式
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.mention_pattern = re.compile(r'@\w+')
        self.special_pattern = re.compile(r'[^\w\s]')
        self.whitespace_pattern = re.compile(r'\s+')
        self.reddit_quote_pattern = re.compile(r'^>.*$', re.MULTILINE)
        # Reddit Markdown 表格和格式符号
        self.markdown_table_pattern = re.compile(r'\|.*\|')  # 表格行
        self.markdown_separator_pattern = re.compile(r':?-{2,}:?')  # 表格分隔符 :---, ---, ---:
        self.markdown_format_pattern = re.compile(r'\*{1,2}[^*]+\*{1,2}')  # **bold** *italic*
        self.repeated_chars_pattern = re.compile(r'(.)\1{3,}')  # 重复字符 aaaa, ----
        
    def clean_text(self, text: str) -> str:
        """清理文本"""
        if not isinstance(text, str):
            return ""
        
        # 移除Reddit引用
        text = self.reddit_quote_pattern.sub('', text)
        
        # 移除Markdown表格
        text = self.markdown_table_pattern.sub(' ', text)
        text = self.markdown_separator_pattern.sub(' ', text)
        
        # 移除Markdown格式符号
        text = self.markdown_format_pattern.sub(' ', text)
        
        # 移除重复字符模式
        text = self.repeated_chars_pattern.sub(' ', text)
        
        # 移除 HTML 实体 (&amp; &gt; &lt; &#x200b; 等)
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        text = re.sub(r'&#x?[0-9a-fA-F]+;?', ' ', text)
        text = re.sub(r'x[0-9a-fA-F]{4}', ' ', text)  # x200b 等
        
        # 移除URL
        if self.config.get("remove_urls", True):
            text = self.url_pattern.sub(' ', text)
        
        # 移除@mentions
        if self.config.get("remove_mentions", True):
            text = self.mention_pattern.sub(' ', text)
        
        # 转小写
        if self.config.get("lowercase", True):
            text = text.lower()
        
        # 移除特殊字符(保留基本标点)
        text = re.sub(r'[^a-zA-Z0-9\s\'\-]', ' ', text)
        
        # 规范化空白
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """分词"""
        text = self.clean_text(text)
        tokens = text.split()
        
        # 移除停用词和无意义token
        if self.config.get("remove_stopwords", True):
            filtered_tokens = []
            for t in tokens:
                # 跳过停用词
                if t in self.stopwords:
                    continue
                # 跳过太短的token
                if len(t) <= 1:
                    continue
                # 跳过纯数字
                if t.isdigit():
                    continue
                # 跳过只包含特殊字符的token (--- , *** , ===)
                if re.match(r'^[\-\*\=\_\|\:]+$', t):
                    continue
                # 跳过包含过多重复字符的token
                if re.search(r'(.)\1{2,}', t):
                    continue
                filtered_tokens.append(t)
            tokens = filtered_tokens
        
        return tokens
    
    def get_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """提取n-gram"""
        if len(tokens) < n:
            return []
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def get_all_ngrams(self, tokens: List[str], ngram_range: Tuple[int, int] = (1, 3)) -> List[str]:
        """提取指定范围的所有n-gram"""
        all_ngrams = []
        for n in range(ngram_range[0], ngram_range[1] + 1):
            all_ngrams.extend(self.get_ngrams(tokens, n))
        return all_ngrams
    
    def get_shingles(self, text: str, k: int = 5) -> set:
        """获取字符级k-shingle集合(用于MinHash)"""
        text = self.clean_text(text)
        if len(text) < k:
            return {text} if text else set()
        return {text[i:i+k] for i in range(len(text) - k + 1)}


class RedditDataLoader:
    """Reddit数据加载器"""
    
    def __init__(self, data_path: str = None):
        """
        初始化数据加载器
        
        Args:
            data_path: CSV文件路径。如果不提供，将在Config.DATA_DIR中查找
        """
        self.data_path = data_path
        self.preprocessor = TextPreprocessor()
        self.df = None
        self.subreddits = []
        
    def find_data_file(self) -> str:
        """查找数据文件"""
        if self.data_path and os.path.exists(self.data_path):
            return self.data_path
        
        # 在data目录中查找
        data_dir = Config.DATA_DIR
        if os.path.exists(data_dir):
            # 优先查找配置的文件
            config_file = os.path.join(data_dir, Config.REDDIT_DATA_FILE)
            if os.path.exists(config_file):
                return config_file
            
            # 查找其他CSV文件
            for f in os.listdir(data_dir):
                if f.endswith('.csv') and ('reddit' in f.lower() or 'kaggle' in f.lower() or 'RC' in f):
                    return os.path.join(data_dir, f)
        
        return None
    
    def load(self, 
             num_comments: int = None,
             subreddits: List[str] = None,
             min_score: int = None,
             random_sample: bool = True) -> pd.DataFrame:
        """
        加载Reddit评论数据 (优化版：全量数据快速加载)
        
        Args:
            num_comments: 加载的评论数量(None表示全部)
            subreddits: 指定的subreddit列表
            min_score: 最小评分过滤
            random_sample: 是否随机采样
            
        Returns:
            处理后的DataFrame
        """
        print("\n📂 加载Reddit数据...")
        
        # 查找数据文件
        data_file = self.find_data_file()
        if not data_file:
            print("⚠️ 未找到Reddit数据文件，将生成模拟数据")
            return self._generate_sample_data(num_comments or 10000)
        
        print(f"   文件: {data_file}")
        
        # 加载CSV
        try:
            # 先读取一小部分确定列名
            sample_df = pd.read_csv(data_file, nrows=5)
            print(f"   列名: {list(sample_df.columns)}")
            
            # 确定要读取的行数
            if num_comments:
                if random_sample and num_comments < 500000:
                    # 小规模采样时随机
                    total_rows = sum(1 for _ in open(data_file, 'r', encoding='utf-8', errors='ignore')) - 1
                    print(f"   总行数: {total_rows:,}")
                    
                    skip_rows = sorted(np.random.choice(
                        range(1, total_rows + 1), 
                        size=max(0, total_rows - num_comments), 
                        replace=False
                    ))
                    self.df = pd.read_csv(data_file, skiprows=skip_rows)
                else:
                    # 大规模直接读取前N行
                    self.df = pd.read_csv(data_file, nrows=num_comments)
            else:
                # 全量数据：使用优化参数快速加载
                print("   全量加载中...")
                self.df = pd.read_csv(
                    data_file,
                    dtype={'subreddit': 'category', 'controversiality': 'int8'},  # 优化内存
                    low_memory=False
                )
            
            print(f"   加载: {len(self.df):,} 条评论")
            
        except Exception as e:
            print(f"⚠️ 加载失败: {e}")
            print("   将生成模拟数据")
            return self._generate_sample_data(num_comments or 10000)
        
        # 标准化列名
        self.df = self._standardize_columns(self.df)
        
        # 过滤
        if subreddits:
            self.df = self.df[self.df['subreddit'].isin(subreddits)]
            print(f"   Subreddit过滤后: {len(self.df):,}")
        
        if min_score is not None:
            self.df = self.df[self.df['score'] >= min_score]
            print(f"   评分过滤后: {len(self.df):,}")
        
        # 预处理 (大数据量时使用并行处理)
        if len(self.df) > 50000:
            self.df = self._preprocess_parallel(self.df)
        else:
            self.df = self._preprocess(self.df)
        
        # 统计信息
        self.subreddits = self.df['subreddit'].unique().tolist()
        print(f"   Subreddits: {len(self.subreddits)}")
        print(f"   有效评论: {len(self.df):,}")
        
        return self.df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        # 常见的列名映射
        column_mapping = {
            'body': 'text',
            'comment': 'text',
            'content': 'text',
            'comment_body': 'text',
            'selftext': 'text',
            'ups': 'score',
            'upvotes': 'score',
            'points': 'score',
            'created': 'timestamp',
            'created_utc': 'timestamp',
            'date': 'timestamp',
            'author': 'author',
            'user': 'author',
            'username': 'author',
            'sub': 'subreddit',
            'sr': 'subreddit',
        }
        
        # 转小写并映射
        df.columns = df.columns.str.lower()
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # 确保必要列存在
        if 'text' not in df.columns:
            # 尝试找到文本列
            text_cols = [c for c in df.columns if 'text' in c or 'body' in c or 'comment' in c]
            if text_cols:
                df['text'] = df[text_cols[0]]
            else:
                df['text'] = df.iloc[:, 0].astype(str)
        
        if 'subreddit' not in df.columns:
            df['subreddit'] = 'unknown'
        
        if 'score' not in df.columns:
            df['score'] = 0
        
        if 'id' not in df.columns:
            df['id'] = range(len(df))
        
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.Timestamp.now()
        
        return df
    
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理数据 (优化版本：批量处理)"""
        print("   预处理中...")
        
        # 移除空评论
        df = df.dropna(subset=['text'])
        df = df[df['text'].str.len() > 0]
        
        # 长度过滤
        min_len = Config.PREPROCESSING.get('min_comment_length', 10)
        max_len = Config.PREPROCESSING.get('max_comment_length', 5000)
        df = df[df['text'].str.len().between(min_len, max_len)]
        
        # 移除[deleted]和[removed]
        df = df[~df['text'].str.contains(r'^\[deleted\]$|^\[removed\]$', regex=True, na=False)]
        
        # 批量预处理 - 使用向量化操作
        print("   批量清理文本...")
        df['clean_text'] = df['text'].apply(self.preprocessor.clean_text)
        
        print("   批量分词...")
        df['tokens'] = df['clean_text'].apply(lambda x: self.preprocessor.tokenize(x))
        df['token_count'] = df['tokens'].apply(len)
        
        # 移除token过少的评论
        df = df[df['token_count'] >= 3]
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        return df
    
    def _preprocess_parallel(self, df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
        """
        并行预处理数据 (大数据量优化版)
        
        Args:
            df: 原始DataFrame
            n_jobs: 并行任务数，-1表示使用所有CPU核心
        """
        print("   并行预处理中...")
        
        # 移除空评论
        df = df.dropna(subset=['text'])
        df = df[df['text'].str.len() > 0]
        
        # 长度过滤
        min_len = Config.PREPROCESSING.get('min_comment_length', 10)
        max_len = Config.PREPROCESSING.get('max_comment_length', 5000)
        df = df[df['text'].str.len().between(min_len, max_len)]
        
        # 移除[deleted]和[removed]
        df = df[~df['text'].str.contains(r'^\[deleted\]$|^\[removed\]$', regex=True, na=False)]
        
        total = len(df)
        print(f"   过滤后: {total:,} 条待处理")
        
        # 对于大数据集，使用批量处理而非完全并行（避免内存问题）
        batch_size = 50000
        all_clean_texts = []
        all_tokens = []
        
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_df = df.iloc[batch_start:batch_end]
            
            # 批量处理文本
            batch_texts = batch_df['text'].tolist()
            batch_clean = [self.preprocessor.clean_text(t) for t in batch_texts]
            batch_tokens = [self.preprocessor.tokenize(t) for t in batch_clean]
            
            all_clean_texts.extend(batch_clean)
            all_tokens.extend(batch_tokens)
            
            print(f"      进度: {batch_end:,}/{total:,} ({100*batch_end/total:.1f}%)")
        
        df = df.copy()
        df['clean_text'] = all_clean_texts
        df['tokens'] = all_tokens
        df['token_count'] = df['tokens'].apply(len)
        
        # 移除token过少的评论
        df = df[df['token_count'] >= 3]
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        return df
        
        # 移除空评论
        df = df.dropna(subset=['text'])
        df = df[df['text'].str.len() > 0]
        
        # 长度过滤
        min_len = Config.PREPROCESSING.get('min_comment_length', 10)
        max_len = Config.PREPROCESSING.get('max_comment_length', 5000)
        df = df[df['text'].str.len().between(min_len, max_len)]
        
        # 移除[deleted]和[removed]
        df = df[~df['text'].str.contains(r'^\[deleted\]$|^\[removed\]$', regex=True, na=False)]
        
        # 添加处理后的文本列
        df['clean_text'] = df['text'].apply(self.preprocessor.clean_text)
        df['tokens'] = df['clean_text'].apply(lambda x: self.preprocessor.tokenize(x))
        df['token_count'] = df['tokens'].apply(len)
        
        # 移除token过少的评论
        df = df[df['token_count'] >= 3]
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        return df
    
    def _generate_sample_data(self, num_comments: int) -> pd.DataFrame:
        """生成模拟数据(当真实数据不可用时)"""
        print(f"   生成 {num_comments:,} 条模拟评论...")
        
        # 模拟的subreddits
        subreddits = [
            'technology', 'politics', 'gaming', 'movies', 'news',
            'worldnews', 'science', 'AskReddit', 'funny', 'pics'
        ]
        
        # 话题模板
        topics = {
            'technology': [
                "AI is going to change everything in the next few years",
                "This new smartphone is actually pretty impressive",
                "I don't trust tech companies with my data anymore",
                "The future of computing is quantum",
                "Self-driving cars are still not ready for mainstream",
                "Open source software is the way to go",
                "This is just another tech bubble waiting to burst",
            ],
            'politics': [
                "The government needs to do more about this issue",
                "I can't believe they passed that bill",
                "Both sides are missing the point here",
                "This is exactly what I voted for",
                "Politicians never keep their promises",
                "We need more transparency in government",
            ],
            'gaming': [
                "This game is absolutely amazing, best I've played",
                "The graphics are insane but gameplay is meh",
                "Can't believe they're charging full price for this",
                "The community for this game is so toxic",
                "I've been playing for 500 hours, no regrets",
                "This is just a reskin of the previous game",
            ],
            'movies': [
                "Best movie I've seen this year, hands down",
                "The plot twist was so predictable",
                "Great acting but terrible writing",
                "This deserves all the awards",
                "I don't understand the hype around this movie",
                "The original was way better",
            ],
            'news': [
                "This is a developing story, stay tuned",
                "Can we get a reliable source for this?",
                "Not surprised, saw this coming",
                "This affects more people than you'd think",
                "Media is blowing this out of proportion",
            ],
        }
        
        # 通用回复模板
        generic_replies = [
            "I completely agree with this",
            "This is so wrong on many levels",
            "Can confirm, I've experienced the same thing",
            "Source?",
            "This deserves more upvotes",
            "Underrated comment right here",
            "First time I've heard about this",
            "Thanks for sharing your perspective",
            "I used to think this way but changed my mind",
            "This is exactly what I was thinking",
            "People really need to understand this better",
            "I disagree but I see where you're coming from",
            "This needs to be higher up",
            "Same here, it's frustrating",
            "Great point, never thought about it that way",
        ]
        
        data = []
        np.random.seed(42)
        
        for i in range(num_comments):
            subreddit = np.random.choice(subreddits)
            
            # 70% 来自话题模板, 30% 通用回复
            if np.random.random() < 0.7 and subreddit in topics:
                base_text = np.random.choice(topics.get(subreddit, topics['technology']))
            else:
                base_text = np.random.choice(generic_replies)
            
            # 添加一些变化
            variations = [
                "",
                " honestly",
                " tbh",
                " imo",
                " lol",
                "!",
                "...",
                " definitely",
            ]
            text = base_text + np.random.choice(variations)
            
            # 15% 重复评论(模拟复读机)
            if np.random.random() < 0.15 and len(data) > 0:
                text = np.random.choice(data)['text']
                # 可能有轻微修改
                if np.random.random() < 0.5:
                    text = text.replace('.', '!').replace(',', '')
            
            data.append({
                'id': i,
                'text': text,
                'subreddit': subreddit,
                'score': int(np.random.exponential(10)),
                'author': f'user_{np.random.randint(1, 10000)}',
                'timestamp': pd.Timestamp.now() - pd.Timedelta(hours=np.random.randint(0, 168)),
            })
        
        df = pd.DataFrame(data)
        df = self._preprocess(df)
        
        self.subreddits = subreddits
        self.df = df
        
        return df
    
    def get_comments_by_subreddit(self, subreddit: str) -> pd.DataFrame:
        """获取特定subreddit的评论"""
        if self.df is None:
            raise ValueError("请先调用load()加载数据")
        return self.df[self.df['subreddit'] == subreddit]
    
    def get_subreddit_stats(self) -> pd.DataFrame:
        """获取各subreddit的统计信息"""
        if self.df is None:
            raise ValueError("请先调用load()加载数据")
        
        stats = self.df.groupby('subreddit').agg({
            'id': 'count',
            'score': ['mean', 'sum'],
            'token_count': 'mean',
        }).round(2)
        
        stats.columns = ['count', 'avg_score', 'total_score', 'avg_tokens']
        stats = stats.sort_values('count', ascending=False)
        
        return stats
    
    def iterate_batches(self, batch_size: int = 10000) -> Generator[pd.DataFrame, None, None]:
        """批量迭代数据(用于大规模处理)"""
        if self.df is None:
            raise ValueError("请先调用load()加载数据")
        
        for i in range(0, len(self.df), batch_size):
            yield self.df.iloc[i:i+batch_size]
    
    def to_documents(self) -> List[Dict]:
        """转换为文档列表格式"""
        if self.df is None:
            raise ValueError("请先调用load()加载数据")
        
        return self.df.to_dict('records')
