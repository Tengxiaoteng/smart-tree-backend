-- 为用户添加可扩展资料表（基本信息 + 树苗画像）
-- MySQL 版本（SQLite 会在首次 create_all 时自动创建表）

CREATE TABLE IF NOT EXISTS user_profile (
  userId VARCHAR(191) NOT NULL PRIMARY KEY,
  email VARCHAR(255) NULL,
  avatarUrl VARCHAR(1024) NULL,
  bio TEXT NULL,
  timezone VARCHAR(64) NULL,
  language VARCHAR(32) NULL,
  education JSON NULL,
  preferences JSON NULL,
  learningHabits JSON NULL,
  seedlingPortrait JSON NULL,
  portraitUpdatedAt DATETIME NULL,
  createdAt DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updatedAt DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT fk_user_profile_user FOREIGN KEY (userId) REFERENCES user(id) ON DELETE CASCADE
);

-- 查看表结构
DESCRIBE user_profile;
