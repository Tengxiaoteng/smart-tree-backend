-- 添加答题记录的追问对话字段
-- 用于存储用户答题后与 AI 的追问对话历史

-- MySQL 版本
ALTER TABLE answerrecord ADD COLUMN followUpMessages JSON NULL;

-- SQLite 版本（如果使用 SQLite）
-- ALTER TABLE answerrecord ADD COLUMN followUpMessages TEXT NULL;

