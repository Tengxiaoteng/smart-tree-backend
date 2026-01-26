-- 为帖子添加 shareCode 字段，替代 topicId
-- 这样其他用户可以通过分享码获取知识树，而不是私有的 topicId

-- 添加 shareCode 字段
ALTER TABLE post ADD COLUMN shareCode VARCHAR(20) NULL;

-- 创建索引
CREATE INDEX idx_post_share_code ON post(shareCode);

-- 注意：topicId 字段保留用于向后兼容，但前端将使用 shareCode

