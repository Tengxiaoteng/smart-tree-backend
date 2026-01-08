-- Add composite indexes to speed up common queries.
-- If an index already exists, you may need to drop it first or ignore the error.

-- KnowledgeNode: (userId, topicId) and (userId, parentId)
CREATE INDEX idx_user_topic ON knowledgenode(userId, topicId);
CREATE INDEX idx_user_parent ON knowledgenode(userId, parentId);

-- Material: (userId, topicId)
CREATE INDEX idx_material_user_topic ON material(userId, topicId);

