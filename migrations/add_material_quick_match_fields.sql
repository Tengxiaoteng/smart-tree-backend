-- 添加资料的快速匹配字段
-- 用于重复资料检测优化

-- MySQL 版本
ALTER TABLE material ADD COLUMN contentDigest VARCHAR(255) NULL;
ALTER TABLE material ADD COLUMN keyTopics JSON NULL;
ALTER TABLE material ADD COLUMN contentHash VARCHAR(64) NULL;
ALTER TABLE material ADD COLUMN digestGeneratedAt DATETIME NULL;

-- 为 contentHash 添加索引以加速重复检测
CREATE INDEX idx_material_content_hash ON material(contentHash);

-- SQLite 版本（如果使用 SQLite）
-- ALTER TABLE material ADD COLUMN contentDigest TEXT NULL;
-- ALTER TABLE material ADD COLUMN keyTopics TEXT NULL;
-- ALTER TABLE material ADD COLUMN contentHash TEXT NULL;
-- ALTER TABLE material ADD COLUMN digestGeneratedAt TEXT NULL;
-- CREATE INDEX idx_material_content_hash ON material(contentHash);

