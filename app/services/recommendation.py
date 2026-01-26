"""
社区帖子个性化推荐服务
基于 TF-IDF + 余弦相似度实现
"""

import logging
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
import re

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# 尝试导入 scikit-learn 和 jieba，如果失败则使用简单匹配
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn 未安装，将使用简单关键词匹配")

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning("jieba 未安装，将使用简单分词")


def tokenize_text(text: str) -> str:
    """
    对文本进行分词处理
    中文使用 jieba，英文使用空格分词
    """
    if not text:
        return ""
    
    # 清理文本
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
    
    if JIEBA_AVAILABLE:
        # 使用 jieba 分词
        words = jieba.cut(text)
        return ' '.join(words)
    else:
        # 简单分词：按空格和中文字符分割
        return text


def build_user_interest_text(topics: List[dict]) -> str:
    """
    从用户的 Topics 构建兴趣文本
    包含：Topic 名称、描述、关键词、范围
    """
    interest_parts = []
    
    for topic in topics:
        # Topic 名称（权重最高，重复3次）
        name = topic.get('name', '')
        if name:
            interest_parts.extend([name] * 3)
        
        # 描述
        description = topic.get('description', '')
        if description:
            interest_parts.append(description)
        
        # 关键词（权重高，重复2次）
        keywords = topic.get('keywords') or []
        for kw in keywords:
            interest_parts.extend([kw] * 2)
        
        # 范围
        scope = topic.get('scope') or []
        interest_parts.extend(scope)
    
    return ' '.join(interest_parts)


def calculate_relevance_scores(
    posts: List[dict],
    user_interest_text: str,
) -> List[Tuple[str, float]]:
    """
    计算帖子与用户兴趣的相关性分数
    返回: [(post_id, score), ...]
    """
    if not posts or not user_interest_text:
        return [(p['id'], 0.0) for p in posts]
    
    # 构建帖子文本
    post_texts = []
    for post in posts:
        # 标题权重更高（重复2次）
        title = post.get('title', '')
        content = post.get('content', '')
        tags = ' '.join(post.get('tags') or [])
        post_text = f"{title} {title} {content} {tags}"
        post_texts.append(tokenize_text(post_text))
    
    # 用户兴趣文本
    user_text = tokenize_text(user_interest_text)
    
    if SKLEARN_AVAILABLE:
        return _calculate_tfidf_scores(posts, post_texts, user_text)
    else:
        return _calculate_simple_scores(posts, post_texts, user_text)


def _calculate_tfidf_scores(
    posts: List[dict],
    post_texts: List[str],
    user_text: str,
) -> List[Tuple[str, float]]:
    """使用 TF-IDF + 余弦相似度计算分数"""
    try:
        # 合并所有文本用于 TF-IDF 向量化
        all_texts = post_texts + [user_text]
        
        # TF-IDF 向量化
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # 使用 unigram 和 bigram
            min_df=1,
            max_df=0.95,
        )
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # 用户向量是最后一个
        user_vector = tfidf_matrix[-1]
        post_vectors = tfidf_matrix[:-1]
        
        # 计算余弦相似度
        similarities = cosine_similarity(post_vectors, user_vector).flatten()
        
        return [(posts[i]['id'], float(similarities[i])) for i in range(len(posts))]
    
    except Exception as e:
        logger.error(f"TF-IDF 计算失败: {e}")
        return [(p['id'], 0.0) for p in posts]


def _calculate_simple_scores(
    posts: List[dict],
    post_texts: List[str],
    user_text: str,
) -> List[Tuple[str, float]]:
    """简单关键词匹配（备用方案）"""
    user_words = set(user_text.lower().split())
    scores = []

    for i, post in enumerate(posts):
        post_words = set(post_texts[i].lower().split())
        # 计算交集比例
        if user_words:
            overlap = len(user_words & post_words)
            score = overlap / len(user_words)
        else:
            score = 0.0
        scores.append((post['id'], score))

    return scores


def get_recommended_posts(
    db: Session,
    user_id: str,
    posts: List,
    time_decay_days: int = 7,
) -> List:
    """
    获取个性化推荐排序的帖子

    Args:
        db: 数据库会话
        user_id: 用户ID
        posts: 原始帖子列表（SQLAlchemy 模型对象）
        time_decay_days: 时间衰减天数

    Returns:
        排序后的帖子列表
    """
    from app.models import Topic

    if not posts:
        return posts

    # 1. 获取用户的所有 Topics
    user_topics = db.query(Topic).filter(Topic.userId == user_id).all()

    if not user_topics:
        # 用户没有知识树，按时间排序
        logger.info(f"用户 {user_id} 没有知识树，使用默认排序")
        return posts

    # 2. 构建用户兴趣文本
    topics_data = [
        {
            'name': t.name,
            'description': t.description,
            'keywords': t.keywords,
            'scope': t.scope,
        }
        for t in user_topics
    ]
    user_interest_text = build_user_interest_text(topics_data)

    logger.info(f"用户 {user_id} 兴趣关键词: {user_interest_text[:100]}...")

    # 3. 构建帖子数据
    posts_data = [
        {
            'id': p.id,
            'title': p.title,
            'content': p.content,
            'tags': p.tags,
            'createdAt': p.createdAt,
        }
        for p in posts
    ]

    # 4. 计算相关性分数
    relevance_scores = calculate_relevance_scores(posts_data, user_interest_text)
    score_map = {post_id: score for post_id, score in relevance_scores}

    # 5. 计算最终分数（相关性 + 时间衰减）
    now = datetime.utcnow()
    final_scores = {}

    for post in posts:
        relevance = score_map.get(post.id, 0.0)

        # 时间衰减：新帖子加分
        days_old = (now - post.createdAt).days if post.createdAt else 0
        time_bonus = max(0, 1 - days_old / time_decay_days) * 0.3

        # 最终分数 = 相关性 * 0.7 + 时间加成 * 0.3
        final_scores[post.id] = relevance * 0.7 + time_bonus

    # 6. 按分数排序
    sorted_posts = sorted(posts, key=lambda p: final_scores.get(p.id, 0), reverse=True)

    logger.info(f"推荐排序完成，共 {len(sorted_posts)} 篇帖子")

    return sorted_posts

