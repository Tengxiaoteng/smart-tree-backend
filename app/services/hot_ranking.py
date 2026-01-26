"""
热门帖子排序服务
使用48小时滑动窗口算法，基于浏览量、评论数、分享数加权计算
"""
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, case, literal
from typing import List, Dict
import logging

from app.models import Post, Comment
from app.models.post_view import PostView

logger = logging.getLogger(__name__)

# 权重配置
WEIGHT_VIEW = 1.0       # 浏览权重（最高）
WEIGHT_COMMENT = 3.0    # 评论权重（中等，一个评论约等于3次浏览）
WEIGHT_SHARE = 0.5      # 分享权重（最低）

# 时间窗口
HOT_WINDOW_HOURS = 48   # 48小时滑动窗口

# 同一用户评论计数上限
MAX_COMMENTS_PER_USER = 2


def calculate_hot_score(
    view_count_48h: int,
    comment_count_48h: int, 
    share_count: int,
    created_at: datetime
) -> float:
    """
    计算帖子热度分数
    
    公式: score = views * W_view + comments * W_comment + shares * W_share
    
    Args:
        view_count_48h: 48小时内去重浏览量
        comment_count_48h: 48小时内评论数
        share_count: 分享数（使用总分享数，因为分享权重低）
        created_at: 帖子创建时间
    
    Returns:
        热度分数
    """
    base_score = (
        view_count_48h * WEIGHT_VIEW +
        comment_count_48h * WEIGHT_COMMENT +
        share_count * WEIGHT_SHARE
    )
    
    # 时间衰减因子：越新的帖子有轻微加成
    now = datetime.utcnow()
    hours_old = (now - created_at).total_seconds() / 3600
    
    # 48小时内的帖子有加成，超过48小时逐渐衰减
    if hours_old <= 48:
        time_bonus = 1.0 + (48 - hours_old) / 480  # 最多10%加成
    else:
        time_bonus = 1.0
    
    return base_score * time_bonus


def get_hot_posts(db: Session, posts: List[Post]) -> List[Post]:
    """
    对帖子列表按热度排序

    Args:
        db: 数据库会话
        posts: 待排序的帖子列表

    Returns:
        按热度排序后的帖子列表
    """
    if not posts:
        return []

    now = datetime.utcnow()
    window_start = now - timedelta(hours=HOT_WINDOW_HOURS)

    post_ids = [p.id for p in posts]

    # 批量查询48小时内的去重浏览量
    view_counts = db.query(
        PostView.postId,
        func.count(PostView.id).label('count')
    ).filter(
        PostView.postId.in_(post_ids),
        PostView.viewedAt >= window_start
    ).group_by(PostView.postId).all()

    view_count_map = {v.postId: v.count for v in view_counts}

    # 批量查询48小时内的评论数（每用户最多计2次）
    # 先查询每个帖子每个用户的评论数
    user_comment_counts = db.query(
        Comment.postId,
        Comment.userId,
        func.count(Comment.id).label('count')
    ).filter(
        Comment.postId.in_(post_ids),
        Comment.createdAt >= window_start
    ).group_by(Comment.postId, Comment.userId).all()

    # 计算每个帖子的有效评论数（每用户最多计2次）
    comment_count_map: Dict[str, int] = {}
    for row in user_comment_counts:
        post_id = row.postId
        # 每用户最多计 MAX_COMMENTS_PER_USER 次
        effective_count = min(row.count, MAX_COMMENTS_PER_USER)
        comment_count_map[post_id] = comment_count_map.get(post_id, 0) + effective_count

    # 计算每个帖子的热度分数
    post_scores = []
    for post in posts:
        view_count_48h = view_count_map.get(post.id, 0)
        comment_count_48h = comment_count_map.get(post.id, 0)

        score = calculate_hot_score(
            view_count_48h=view_count_48h,
            comment_count_48h=comment_count_48h,
            share_count=post.shareCount or 0,
            created_at=post.createdAt
        )

        post_scores.append((post, score))

        logger.debug(
            f"Post {post.id}: views_48h={view_count_48h}, "
            f"comments_48h={comment_count_48h}, shares={post.shareCount}, "
            f"score={score:.2f}"
        )

    # 按分数降序排序
    post_scores.sort(key=lambda x: x[1], reverse=True)

    return [p for p, _ in post_scores]


def record_post_view(db: Session, post_id: str, user_id: str) -> bool:
    """
    记录用户浏览帖子（去重）
    
    Args:
        db: 数据库会话
        post_id: 帖子ID
        user_id: 用户ID
    
    Returns:
        True 如果是新浏览记录，False 如果已存在
    """
    # 检查是否已有浏览记录
    existing = db.query(PostView).filter(
        PostView.postId == post_id,
        PostView.userId == user_id
    ).first()
    
    if existing:
        # 更新浏览时间（可选，用于追踪最后访问）
        existing.viewedAt = datetime.utcnow()
        db.commit()
        return False
    
    # 创建新浏览记录
    view = PostView(
        postId=post_id,
        userId=user_id
    )
    db.add(view)
    
    # 更新帖子的总浏览量
    post = db.query(Post).filter(Post.id == post_id).first()
    if post:
        post.viewCount = (post.viewCount or 0) + 1
    
    db.commit()
    logger.info(f"Recorded view: user={user_id}, post={post_id}")
    return True

