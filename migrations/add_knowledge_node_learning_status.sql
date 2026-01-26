-- 添加知识节点的学习状态跟踪字段
-- 用于存储用户的学习进度、复习时间、信心等级等信息

-- MySQL 版本
ALTER TABLE knowledgenode ADD COLUMN learningStatus JSON NULL;

-- SQLite 版本（如果使用 SQLite）
-- ALTER TABLE knowledgenode ADD COLUMN learningStatus TEXT NULL;

