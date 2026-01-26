"""
数据库迁移定义

所有迁移按顺序定义，迁移管理器会自动跳过已执行的迁移。
"""

from app.core.database import engine
from app.core.migrations import MigrationManager

# 创建迁移管理器实例
migration_manager = MigrationManager(engine)


# ============ 迁移定义 ============

@migration_manager.register("001_material_ai_fields", "Material 表 AI 相关字段")
def migrate_material_ai_fields(m: MigrationManager):
    """补齐 Material 表的 AI 字段"""
    if not m.table_exists("material"):
        return
    m.add_column("material", "organizedContent", "LONGTEXT NULL", "TEXT")
    m.add_column("material", "aiSummary", "LONGTEXT NULL", "TEXT")
    m.add_column("material", "extractedConcepts", "JSON NULL", "TEXT")
    m.add_column("material", "isOrganized", "TINYINT(1) NOT NULL DEFAULT 0", "BOOLEAN NOT NULL DEFAULT 0")
    m.add_column("material", "structuredContent", "JSON NULL", "TEXT")
    m.add_column("material", "isStructured", "TINYINT(1) NOT NULL DEFAULT 0", "BOOLEAN NOT NULL DEFAULT 0")


@migration_manager.register("002_material_quick_match", "Material 表快速匹配字段")
def migrate_material_quick_match(m: MigrationManager):
    """补齐 Material 表的快速匹配字段"""
    if not m.table_exists("material"):
        return
    m.add_column("material", "contentDigest", "VARCHAR(255) NULL")
    m.add_column("material", "keyTopics", "JSON NULL", "TEXT")
    m.add_column("material", "contentHash", "VARCHAR(64) NULL")
    m.add_column("material", "digestGeneratedAt", "DATETIME NULL")


@migration_manager.register("003_user_profile_email", "UserProfile 表 email 字段")
def migrate_user_profile_email(m: MigrationManager):
    """补齐 UserProfile 表的 email 字段"""
    if not m.table_exists("user_profile"):
        return
    m.add_column("user_profile", "email", "VARCHAR(255) NULL")


@migration_manager.register("004_question_fields", "Question 表新字段")
def migrate_question_fields(m: MigrationManager):
    """补齐 Question 表的新字段"""
    if not m.table_exists("question"):
        return
    m.add_column("question", "hints", "JSON NULL", "TEXT")
    m.add_column("question", "relatedConcepts", "JSON NULL", "TEXT")
    m.add_column("question", "parentQuestionId", "VARCHAR(191) NULL")
    m.add_column("question", "derivationType", "VARCHAR(255) NULL")


@migration_manager.register("005_answerrecord_followup", "AnswerRecord 表追问字段")
def migrate_answerrecord_followup(m: MigrationManager):
    """补齐 AnswerRecord 表的追问对话字段"""
    if not m.table_exists("answerrecord"):
        return
    m.add_column("answerrecord", "followUpMessages", "JSON NULL", "TEXT")


@migration_manager.register("006_knowledgenode_learning", "KnowledgeNode 表学习状态字段")
def migrate_knowledgenode_learning(m: MigrationManager):
    """补齐 KnowledgeNode 表的学习状态字段"""
    if not m.table_exists("knowledgenode"):
        return
    m.add_column("knowledgenode", "learningStatus", "JSON NULL", "TEXT")


@migration_manager.register("007_composite_indexes", "复合索引优化")
def migrate_composite_indexes(m: MigrationManager):
    """添加复合索引提升查询性能"""
    if m.table_exists("knowledgenode"):
        m.add_index("knowledgenode", "idx_user_topic", ["userId", "topicId"])
        m.add_index("knowledgenode", "idx_user_parent", ["userId", "parentId"])
    if m.table_exists("material"):
        m.add_index("material", "idx_material_user_topic", ["userId", "topicId"])


@migration_manager.register("008_question_indexes", "Question 表索引优化")
def migrate_question_indexes(m: MigrationManager):
    """添加 Question 表的复合索引"""
    if m.table_exists("question"):
        m.add_index("question", "idx_question_user_node", ["userId", "nodeId"])


@migration_manager.register("009_answerrecord_indexes", "AnswerRecord 表索引优化")
def migrate_answerrecord_indexes(m: MigrationManager):
    """添加 AnswerRecord 表的复合索引"""
    if m.table_exists("answerrecord"):
        m.add_index("answerrecord", "idx_answerrecord_user_question", ["userId", "questionId"])


@migration_manager.register("010_post_indexes", "Post 表索引优化")
def migrate_post_indexes(m: MigrationManager):
    """添加 Post 表的复合索引（社区功能）"""
    if m.table_exists("post"):
        m.add_index("post", "idx_post_category_created", ["categoryId", "createdAt"])
        m.add_index("post", "idx_post_user_created", ["userId", "createdAt"])


@migration_manager.register("011_tree_share_tables", "知识树分享相关表")
def migrate_tree_share_tables(m: MigrationManager):
    """创建知识树分享相关的表"""

    # 创建 tree_share 表
    m.create_table("tree_share", """
        id VARCHAR(191) PRIMARY KEY,
        topicId VARCHAR(191) NOT NULL,
        ownerId VARCHAR(191) NOT NULL,
        shareCode VARCHAR(20) NOT NULL UNIQUE,
        shareType VARCHAR(20) NOT NULL DEFAULT 'public',
        sharePassword VARCHAR(255) NULL,
        shareTitle VARCHAR(255) NULL,
        shareDescription TEXT NULL,
        isActive TINYINT(1) NOT NULL DEFAULT 1,
        currentVersion INT NOT NULL DEFAULT 1,
        subscriberCount INT NOT NULL DEFAULT 0,
        copyCount INT NOT NULL DEFAULT 0,
        viewCount INT NOT NULL DEFAULT 0,
        createdAt DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updatedAt DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_tree_share_owner (ownerId),
        INDEX idx_tree_share_topic (topicId),
        UNIQUE INDEX idx_tree_share_code (shareCode),
        FOREIGN KEY (topicId) REFERENCES topic(id) ON DELETE CASCADE,
        FOREIGN KEY (ownerId) REFERENCES user(id) ON DELETE CASCADE
    """)

    # 创建 tree_subscription 表
    m.create_table("tree_subscription", """
        id VARCHAR(191) PRIMARY KEY,
        shareId VARCHAR(191) NOT NULL,
        subscriberId VARCHAR(191) NOT NULL,
        localTopicId VARCHAR(191) NOT NULL,
        syncedVersion INT NOT NULL DEFAULT 1,
        lastSyncedAt DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        autoSync TINYINT(1) NOT NULL DEFAULT 0,
        notifyOnUpdate TINYINT(1) NOT NULL DEFAULT 1,
        nodeMapping LONGTEXT NULL,
        materialMapping LONGTEXT NULL,
        createdAt DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updatedAt DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_tree_subscription_share (shareId),
        INDEX idx_tree_subscription_subscriber (subscriberId),
        INDEX idx_tree_subscription_local_topic (localTopicId),
        UNIQUE INDEX idx_tree_subscription_unique (shareId, subscriberId),
        FOREIGN KEY (shareId) REFERENCES tree_share(id) ON DELETE CASCADE,
        FOREIGN KEY (subscriberId) REFERENCES user(id) ON DELETE CASCADE,
        FOREIGN KEY (localTopicId) REFERENCES topic(id) ON DELETE CASCADE
    """)

    # 创建 tree_version 表
    m.create_table("tree_version", """
        id VARCHAR(191) PRIMARY KEY,
        shareId VARCHAR(191) NOT NULL,
        version INT NOT NULL,
        snapshotData LONGTEXT NOT NULL,
        changeLog TEXT NULL,
        changesSummary TEXT NULL,
        publishedAt DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_tree_version_share (shareId),
        UNIQUE INDEX idx_tree_version_share_version (shareId, version),
        FOREIGN KEY (shareId) REFERENCES tree_share(id) ON DELETE CASCADE
    """)

    # 创建 tree_update_notification 表
    m.create_table("tree_update_notification", """
        id VARCHAR(191) PRIMARY KEY,
        subscriptionId VARCHAR(191) NOT NULL,
        versionId VARCHAR(191) NOT NULL,
        userId VARCHAR(191) NOT NULL,
        isRead TINYINT(1) NOT NULL DEFAULT 0,
        isApplied TINYINT(1) NOT NULL DEFAULT 0,
        createdAt DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        readAt DATETIME NULL,
        appliedAt DATETIME NULL,
        INDEX idx_notification_user (userId),
        INDEX idx_notification_subscription (subscriptionId),
        INDEX idx_notification_version (versionId),
        INDEX idx_notification_user_unread (userId, isRead),
        FOREIGN KEY (subscriptionId) REFERENCES tree_subscription(id) ON DELETE CASCADE,
        FOREIGN KEY (versionId) REFERENCES tree_version(id) ON DELETE CASCADE,
        FOREIGN KEY (userId) REFERENCES user(id) ON DELETE CASCADE
    """)


@migration_manager.register("012_topic_share_fields", "Topic 表分享相关字段")
def migrate_topic_share_fields(m: MigrationManager):
    """为 Topic 表添加分享相关字段"""
    if not m.table_exists("topic"):
        return
    m.add_column("topic", "sourceShareId", "VARCHAR(191) NULL")
    m.add_column("topic", "isShared", "TINYINT(1) NOT NULL DEFAULT 0")
    m.add_column("topic", "originalNodeMapping", "JSON NULL", "TEXT")
    m.add_column("topic", "originalMaterialMapping", "JSON NULL", "TEXT")
    m.add_index("topic", "idx_topic_source_share", ["sourceShareId"])


def run_migrations():
    """运行所有迁移"""
    migration_manager.run_all()

