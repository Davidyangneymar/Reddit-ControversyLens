# HotCommentLens - Reddit评论分析工具

基于大数据方法的Reddit评论TF-IDF关键词提取与争议性分析工具。

## 📊 功能特性

### 1. TF-IDF关键词提取
- 识别各Subreddit的特色关键词
- 自动降低常见词权重，突出社区特有词汇

### 2. 争议性分析
- 统计争议性评论分布
- 对比争议vs非争议评论的特征差异
- 识别最具争议性的Subreddit
- 提取争议性话题关键词

## 🔧 使用的大数据方法

| 方法 | 说明 | 应用场景 |
|------|------|----------|
| **TF-IDF** | Term Frequency - Inverse Document Frequency | 关键词提取，识别社区特色词汇 |
| **MapReduce** | Map: 分词提取 → Reduce: 词频聚合 | 大规模词频统计、分组聚合 |
| **向量化操作** | pandas/numpy替代Python循环 | 提升10-100倍计算性能 |
| **分层抽样** | 按类别比例抽样 | 平衡争议性vs非争议性样本对比 |

### TF-IDF 公式
```
TF-IDF = TF × IDF

TF (词频) = 词在文档中出现次数 / 文档总词数
IDF (逆文档频率) = log(总文档数 / 包含该词的文档数)
```

### MapReduce 流程
```
Map阶段:   文档 → 分词 → [(word, 1), (word, 1), ...]
Reduce阶段: 按组聚合 → {group: {word: count}}
```

## 🚀 快速开始

### 安装依赖
```bash
pip install pandas numpy matplotlib
```

### 运行分析
```bash
# 完整分析 (100万条评论)
python -m HotCommentLens.run_analysis

# 快速测试 (10万条抽样)
python -m HotCommentLens.run_analysis --sample 100000

# 指定数据文件
python -m HotCommentLens.run_analysis --data path/to/data.csv

# 指定输出目录
python -m HotCommentLens.run_analysis --output path/to/output
```

### 命令行参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--sample` | 抽样数量 | 全部数据 |
| `--data` | 数据文件路径 | data/kaggle_RC_2019-05.csv |
| `--output` | 输出目录 | outputs/tfidf_controversy |

## 📁 输出文件

```
outputs/tfidf_controversy/
├── visualizations/
│   ├── tfidf_keywords.png       # TF-IDF关键词图
│   ├── controversy_analysis.png # 争议性统计图
│   └── keyword_comparison.png   # 关键词对比图
├── tfidf_keywords.csv           # 各subreddit TF-IDF关键词
├── controversy_stats.csv        # 争议性统计
├── subreddit_controversy_rates.csv # 各subreddit争议率
└── keyword_comparison.csv       # 争议vs非争议关键词
```

## 📈 分析结果示例

### TF-IDF关键词 (各社区特色词)
- `r/politics`: trump, barr, mueller, report, congress
- `r/gameofthrones`: arya, episode, spoiler
- `r/news`: gun, guns, school
- `r/aww`: cat, dog, cute

### 争议性分析
- 争议性评论占比: **3.12%**
- 最具争议的subreddit: `r/news` (9.90%), `r/worldnews` (8.67%)
- 争议性评论更长 (207字符 vs 184字符)
- 争议性评论得分更低 (0.52 vs 12.42)
- 争议性高频词: venezuela, guns, racist, trump

## 📂 项目结构

```
HotCommentLens/
├── config.py            # 配置文件
├── data_loader.py       # 数据加载与预处理
├── tfidf_controversy.py # TF-IDF + 争议性分析核心模块
├── run_analysis.py      # 运行入口
└── README.md            # 说明文档
```

## 📊 数据集

使用 Kaggle Reddit 评论数据集:
- 来源: [1 Million Reddit Comments from 40 Subreddits](https://www.kaggle.com/datasets/smagnan/1-million-reddit-comments-from-40-subreddits)
- 数据量: 100万条评论
- 字段: subreddit, body, controversiality, score
- 时间: 2019年5月

## ⏱️ 性能

| 数据量 | TF-IDF分析 | 争议性分析 | 总耗时 |
|--------|-----------|-----------|--------|
| 10万条 | ~2秒 | ~1秒 | ~23秒 |
| 100万条 | ~29秒 | ~3秒 | ~3.5分钟 |

## 📜 License

MIT License
