from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional


# ============ Category Schemas ============
class CategoryBase(BaseModel):
    name: str
    icon: Optional[str] = None
    sortOrder: int = 0


class CategoryCreate(CategoryBase):
    id: Optional[str] = None


class CategoryResponse(CategoryBase):
    id: str

    model_config = ConfigDict(from_attributes=True)


# ============ Post Schemas ============
class PostBase(BaseModel):
    title: str
    content: str
    images: Optional[list[str]] = None
    nodeId: Optional[str] = None
    topicId: Optional[str] = None  # 已废弃，保留向后兼容
    shareCode: Optional[str] = None  # 关联的分享码
    categoryId: Optional[str] = None
    tags: Optional[list[str]] = None


class PostCreate(PostBase):
    pass


class PostUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    images: Optional[list[str]] = None
    nodeId: Optional[str] = None
    topicId: Optional[str] = None  # 已废弃
    shareCode: Optional[str] = None  # 关联的分享码
    categoryId: Optional[str] = None
    tags: Optional[list[str]] = None


class PostAuthor(BaseModel):
    id: str
    username: str
    nickname: Optional[str] = None
    avatarUrl: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class PostResponse(PostBase):
    id: str
    userId: str
    viewCount: int
    commentCount: int
    shareCount: int
    createdAt: datetime
    updatedAt: datetime
    author: Optional[PostAuthor] = None

    model_config = ConfigDict(from_attributes=True)


class PostListResponse(BaseModel):
    items: list[PostResponse]
    total: int
    page: int
    limit: int
    hasMore: bool


# ============ Comment Schemas ============
class CommentBase(BaseModel):
    content: str
    parentId: Optional[str] = None
    replyToUserId: Optional[str] = None


class CommentCreate(CommentBase):
    pass


class CommentAuthor(BaseModel):
    id: str
    username: str
    nickname: Optional[str] = None
    avatarUrl: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class CommentResponse(CommentBase):
    id: str
    postId: str
    userId: str
    createdAt: datetime
    author: Optional[CommentAuthor] = None
    replyToUser: Optional[CommentAuthor] = None
    replies: Optional[list["CommentResponse"]] = None

    model_config = ConfigDict(from_attributes=True)


# ============ Share Schema ============
class ShareResponse(BaseModel):
    postId: str
    title: str
    url: str
    shareText: str
