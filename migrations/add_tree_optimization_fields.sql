-- 数据库优化：为知识树添加辅助字段
-- 这些字段可以加快树形结构的查询和显示

-- 1. 添加 level 字段（节点层级，根节点为 0）
ALTER TABLE knowledgenode ADD COLUMN level INT DEFAULT 0 COMMENT '节点层级，根节点为0';

-- 2. 添加 path 字段（从根到当前节点的路径）
ALTER TABLE knowledgenode ADD COLUMN path VARCHAR(1000) DEFAULT '/' COMMENT '节点路径，例如 /root/child1/grandchild1';

-- 3. 添加 orderIndex 字段（同级节点的排序）
ALTER TABLE knowledgenode ADD COLUMN orderIndex INT DEFAULT 0 COMMENT '同级节点的排序索引';

-- 4. 为新字段添加索引
CREATE INDEX idx_knowledgenode_level ON knowledgenode(level);
CREATE INDEX idx_knowledgenode_path ON knowledgenode(path);

-- 5. 更新现有数据的 level 和 path
-- 注意：这个需要递归更新，建议用 Python 脚本处理

-- 查看表结构
DESCRIBE knowledgenode;

