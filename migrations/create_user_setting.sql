-- Create table: user_setting
-- MySQL / MariaDB migration for Smart Tree (FastAPI backend)

CREATE TABLE IF NOT EXISTS `user_setting` (
  `id` varchar(191) NOT NULL,
  `userId` varchar(191) NOT NULL,
  `apiKey` longtext NULL,
  `modelId` varchar(255) NULL,
  `baseUrl` varchar(1024) NULL,
  `extras` json NULL,
  `createdAt` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updatedAt` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uniq_user_setting_userId` (`userId`),
  CONSTRAINT `fk_user_setting_user` FOREIGN KEY (`userId`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
