-- Create tables: user_credit_account / user_credit_ledger
-- MySQL / MariaDB migration for Credits (points ledger with HMAC chain)

CREATE TABLE IF NOT EXISTS `user_credit_account` (
  `userId` varchar(191) NOT NULL,
  `balance` int NOT NULL DEFAULT 0,
  `lastLedgerSig` varchar(64) NOT NULL DEFAULT '',
  `createdAt` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updatedAt` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`userId`),
  CONSTRAINT `fk_user_credit_account_user` FOREIGN KEY (`userId`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `user_credit_ledger` (
  `id` varchar(191) NOT NULL,
  `userId` varchar(191) NOT NULL,
  `kind` varchar(64) NOT NULL,
  `delta` int NOT NULL,
  `balanceAfter` int NOT NULL,
  `requestId` varchar(191) NULL,
  `model` varchar(255) NULL,
  `promptTokens` int NULL,
  `completionTokens` int NULL,
  `totalTokens` int NULL,
  `costRmbMilli` int NULL,
  `meta` json NULL,
  `prevSig` varchar(64) NOT NULL DEFAULT '',
  `sig` varchar(64) NOT NULL DEFAULT '',
  `createdAt` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_user_credit_ledger_user_created` (`userId`, `createdAt`),
  UNIQUE KEY `uniq_user_credit_ledger_request_kind` (`userId`, `requestId`, `kind`),
  CONSTRAINT `fk_user_credit_ledger_user` FOREIGN KEY (`userId`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

