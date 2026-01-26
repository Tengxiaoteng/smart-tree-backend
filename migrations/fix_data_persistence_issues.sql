-- =====================================================
-- 数据库表设计修复迁移脚本
-- 修复前后端数据模型不一致的问题
-- 执行时间: 2026-01-08
-- =====================================================

-- 1. 添加答题记录的追问对话字段
-- 用于存储用户答题后与 AI 的追问对话历史
ALTER TABLE answerrecord ADD COLUMN IF NOT EXISTS followUpMessages JSON NULL;

-- 2. 添加知识节点的学习状态跟踪字段
-- 用于存储用户的学习进度、复习时间、信心等级等信息
ALTER TABLE knowledgenode ADD COLUMN IF NOT EXISTS learningStatus JSON NULL;

-- 3. 添加资料的快速匹配字段
-- 用于重复资料检测优化
ALTER TABLE material ADD COLUMN IF NOT EXISTS contentDigest VARCHAR(255) NULL;
ALTER TABLE material ADD COLUMN IF NOT EXISTS keyTopics JSON NULL;
ALTER TABLE material ADD COLUMN IF NOT EXISTS contentHash VARCHAR(64) NULL;
ALTER TABLE material ADD COLUMN IF NOT EXISTS digestGeneratedAt DATETIME NULL;

-- 为 contentHash 添加索引以加速重复检测（如果不存在）
-- MySQL 8.0+ 语法
CREATE INDEX IF NOT EXISTS idx_material_content_hash ON material(contentHash);

-- =====================================================
-- 注意事项:
-- 1. UserSettings 的 useSystemKey 和 routing 字段已通过 extras JSON 字段存储
--    无需额外添加列，后端已正确处理
-- 2. 执行前请备份数据库
-- 3. 如果使用 MySQL 5.7，需要移除 IF NOT EXISTS 语法
-- =====================================================

