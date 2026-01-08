-- Create table: user_batch_job
-- MySQL / MariaDB migration for DashScope Batch jobs

CREATE TABLE IF NOT EXISTS `user_batch_job` (
  `id` varchar(191) NOT NULL,
  `userId` varchar(191) NOT NULL,
  `provider` varchar(64) NOT NULL DEFAULT 'dashscope',
  `mode` varchar(16) NOT NULL,
  `baseUrl` varchar(1024) NOT NULL,
  `endpoint` varchar(255) NOT NULL,
  `model` varchar(255) NOT NULL,
  `completionWindow` varchar(32) NOT NULL DEFAULT '24h',
  `batchId` varchar(255) NOT NULL,
  `inputFileId` varchar(255) NOT NULL,
  `outputFileId` varchar(255) NULL,
  `errorFileId` varchar(255) NULL,
  `status` varchar(32) NOT NULL DEFAULT 'validating',
  `requestCounts` json NULL,
  `metadata` json NULL,
  `providerData` json NULL,
  `reservedPoints` int NULL,
  `chargedPoints` int NULL,
  `promptTokens` int NULL,
  `completionTokens` int NULL,
  `totalTokens` int NULL,
  `costRmbMilli` int NULL,
  `billedAt` datetime NULL,
  `createdAt` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updatedAt` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uniq_user_batch_job_batchId` (`batchId`),
  KEY `idx_user_batch_job_userId` (`userId`),
  CONSTRAINT `fk_user_batch_job_user` FOREIGN KEY (`userId`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

