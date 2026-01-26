-- 为 material 表添加 AI 整理/结构化字段（用于“AI 整理 / 出题化”后持久化）

ALTER TABLE material ADD COLUMN organizedContent LONGTEXT NULL COMMENT 'AI 整理后的 Markdown 内容';
ALTER TABLE material ADD COLUMN aiSummary LONGTEXT NULL COMMENT 'AI 摘要';
ALTER TABLE material ADD COLUMN extractedConcepts JSON NULL COMMENT 'AI 提取概念列表';
ALTER TABLE material ADD COLUMN isOrganized TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否已 AI 整理';
ALTER TABLE material ADD COLUMN structuredContent JSON NULL COMMENT '结构化内容（知识点/可出题点）';
ALTER TABLE material ADD COLUMN isStructured TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否已结构化可出题';

-- 查看表结构
DESCRIBE material;

