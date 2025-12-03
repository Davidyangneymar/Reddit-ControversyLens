"""
HotCommentLens 配置文件
"""

import os
from typing import Dict, List, Any

class Config:
    """全局配置类"""
    
    # ==================== 数据路径配置 ====================
    # Reddit 数据集路径 (Kaggle: 1-million-reddit-comments-from-40-subreddits)
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    REDDIT_DATA_FILE = "kaggle_RC_2019-05.csv"  # Kaggle Reddit 数据集
    
    # 输出目录
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    
    # ==================== 文本预处理配置 ====================
    PREPROCESSING = {
        "min_comment_length": 10,      # 最小评论长度(字符)
        "max_comment_length": 5000,    # 最大评论长度
        "remove_urls": True,           # 移除URL
        "remove_mentions": True,       # 移除@mentions
        "lowercase": True,             # 转小写
        "remove_stopwords": True,      # 移除停用词
        "lemmatize": False,            # 词形还原(较慢)
    }
    
    # 停用词列表
    STOPWORDS = {
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
        "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
        'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
        "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
        'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
        'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
        'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
        "wouldn't", 'also', 'would', 'could', 'get', 'got', 'like', 'one',
        'even', 'really', 'think', 'know', 'make', 'want', 'go', 'going',
        'see', 'thing', 'things', 'way', 'people', 'much', 'well', 'back',
        'still', 'something', 'anything', 'nothing', 'everything', 'someone',
        'anyone', 'everyone', 'something', 'yeah', 'yes', 'no', 'ok', 'okay',
        'im', 'ive', 'id', 'youre', 'youve', 'hes', 'shes', 'its', 'were',
        'theyre', 'weve', 'theyve', 'dont', 'doesnt', 'didnt', 'cant', 'wont',
        'shouldnt', 'wouldnt', 'couldnt', 'isnt', 'arent', 'wasnt', 'werent',
        'hasnt', 'havent', 'hadnt', 'thats', 'whats', 'hows', 'whys', 'wheres',
    }
    
    # Reddit 特定停用词
    REDDIT_STOPWORDS = {
        'reddit', 'sub', 'subreddit', 'post', 'comment', 'comments', 'thread',
        'op', 'edit', 'update', 'deleted', 'removed', 'bot', 'mod', 'mods',
        'upvote', 'upvotes', 'downvote', 'downvotes', 'karma', 'gold', 'award',
        'link', 'source', 'sauce', 'repost', 'crosspost', 'xpost', 'tl', 'dr',
        'tldr', 'ama', 'eli5', 'iirc', 'afaik', 'imho', 'imo', 'ftfy', 'til',
        'lol', 'lmao', 'rofl', 'omg', 'wtf', 'smh', 'tbh', 'ngl', 'idk',
        # Reddit Markdown 语法噪声
        'gt', 'lt', 'amp', 'nbsp', 'http', 'https', 'www', 'com', 'org', 'net',
        # 常见无意义词
        'just', 'like', 'dont', 'didnt', 'doesnt', 'thats', 'youre', 'theyre',
        'gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'cause', 'cuz', 'tho',
        # 常见机器人/自动回复词
        'submission', 'submitting', 'automatically', 'removed', 'reason',
        'moderator', 'moderators', 'rule', 'rules', 'violation', 'please',
        'message', 'contact', 'questions', 'concerns',
        # 表格分隔符和格式符号
        'trader', 'karma', 'cake', 'day', 'upvotes', 'posts', 'ign', 'set',
        # HTML 实体和编码
        'x200b', 'x200d', 'x2019', 'x2018', 'x201c', 'x201d', 'quot', 'apos',
        # 常见缩写和噪声
        'ta', 'bc', 'rn', 'af', 'asap', 'fyi', 'btw', 'irl', 'jk', 'nvm',
        'pls', 'plz', 'thx', 'ty', 'yw', 'np', 'ofc', 'obv', 'prob', 'def',
    }
    
    # ==================== 热点检测配置 ====================
    HOTSPOT = {
        "ngram_range": (1, 3),          # N-gram范围: unigram到trigram
        "min_doc_freq": 5,              # 最小文档频率
        "max_doc_freq_ratio": 0.5,      # 最大文档频率比例
        "top_k_hotspots": 10,           # 返回Top-K热点
        "burst_threshold": 2.0,         # 突发度阈值(相对于基线的倍数)
        "use_tfidf": True,              # 使用TF-IDF加权
        "time_window_hours": 24,        # 时间窗口(小时)
    }
    
    # ==================== MinHash/LSH 配置 ====================
    MINHASH = {
        "num_permutations": 128,        # MinHash签名维度
        "shingle_size": 5,              # Shingle大小(字符级5-gram)
        "num_bands": 16,                # LSH bands数量
        "rows_per_band": 8,             # 每个band的行数
        # 近似阈值: t ≈ (1/b)^(1/r) = (1/16)^(1/8) ≈ 0.54
    }
    
    # ==================== 重复过滤配置 ====================
    DUPLICATE = {
        "similarity_threshold": 0.5,    # Jaccard相似度阈值
        "verify_candidates": True,      # 是否精确验证候选对
        "fold_strategy": "first",       # 折叠策略: first/longest/highest_score
    }
    
    # ==================== 观点聚类配置 ====================
    CLUSTERING = {
        "method": "kmeans",             # 聚类方法: kmeans/hierarchical/dbscan
        "vectorizer": "tfidf",          # 向量化方法: tfidf/sbert
        "num_clusters": 5,              # 聚类数量(kmeans)
        "min_cluster_size": 3,          # 最小簇大小
        "sbert_model": "all-MiniLM-L6-v2",  # SBERT模型(如使用)
        "use_umap": True,               # 使用UMAP降维
        "umap_n_neighbors": 15,         # UMAP邻居数
        "umap_min_dist": 0.1,           # UMAP最小距离
    }
    
    # ==================== 可视化配置 ====================
    VISUALIZATION = {
        "figure_dpi": 150,              # 图片DPI
        "color_palette": "Set2",        # 颜色调色板
        "wordcloud_max_words": 100,     # 词云最大词数
        "show_top_n_per_cluster": 3,    # 每个簇显示的代表评论数
    }
    
    # ==================== 运行模式配置 ====================
    RUN_MODES = {
        "demo": {
            "num_comments": 5000,
            "top_k_hotspots": 5,
            "num_clusters": 4,
        },
        "medium": {
            "num_comments": 100000,
            "top_k_hotspots": 10,
            "num_clusters": 5,
        },
        "large": {
            "num_comments": 500000,
            "top_k_hotspots": 15,
            "num_clusters": 8,
        },
        "full": {
            "num_comments": None,  # 全部数据
            "top_k_hotspots": 20,
            "num_clusters": 10,
        }
    }
    
    @classmethod
    def get_all_stopwords(cls) -> set:
        """获取所有停用词"""
        return cls.STOPWORDS | cls.REDDIT_STOPWORDS
    
    @classmethod
    def get_mode_config(cls, mode: str) -> Dict[str, Any]:
        """获取运行模式配置"""
        return cls.RUN_MODES.get(mode, cls.RUN_MODES["demo"])
    
    @classmethod
    def ensure_dirs(cls):
        """确保必要目录存在"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
