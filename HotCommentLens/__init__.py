"""
HotCommentLens: Reddit评论热点检测、重复折叠与观点聚类系统

核心功能:
1. 热点主题检测 (MapReduce)
2. 近重复评论折叠 (MinHash + LSH)
3. 观点聚类 (TF-IDF/Embedding + Clustering)
4. 舆情可视化
"""

__version__ = "1.0.0"
__author__ = "HotCommentLens Team"

from .config import Config
from .data_loader import RedditDataLoader
from .hotspot_detector import HotspotDetector
from .minhash_lsh import MinHashLSH
from .duplicate_filter import DuplicateFilter
from .opinion_clustering import OpinionClusterer
from .pipeline import HotCommentLensPipeline
from .visualization import Visualizer
