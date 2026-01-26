-- 清空trevia数据库所有表数据的SQL脚本
-- 保留表结构，仅删除数据
-- 使用前请确认数据库和表名

-- 禁用外键检查
SET FOREIGN_KEY_CHECKS = 0;

-- 清空所有表数据（按照依赖关系排序）
-- 注意: TRUNCATE 会重置自增ID，DELETE 不会

-- 1. 清空子表（有外键依赖的表）
TRUNCATE TABLE `tree_update_notification`;
TRUNCATE TABLE `tree_version`;
TRUNCATE TABLE `tree_subscription`;
TRUNCATE TABLE `tree_share`;
TRUNCATE TABLE `post_view`;
TRUNCATE TABLE `comment`;
TRUNCATE TABLE `post`;
TRUNCATE TABLE `user_batch_job`;
TRUNCATE TABLE `user_credit_ledger`;
TRUNCATE TABLE `user_credit_account`;
TRUNCATE TABLE `user_note`;
TRUNCATE TABLE `answer_record`;
TRUNCATE TABLE `question`;
TRUNCATE TABLE `user_file`;
TRUNCATE TABLE `material`;
TRUNCATE TABLE `knowledge_node`;
TRUNCATE TABLE `user_profile`;
TRUNCATE TABLE `user_settings`;
TRUNCATE TABLE `category`;
TRUNCATE TABLE `topic`;

-- 2. 清空主表（被其他表引用的表）
TRUNCATE TABLE `user`;

-- 重新启用外键检查
SET FOREIGN_KEY_CHECKS = 1;

-- 验证所有表已清空
SELECT
    TABLE_NAME as '表名',
    TABLE_ROWS as '行数'
FROM
    information_schema.TABLES
WHERE
    TABLE_SCHEMA = 'trevia'
    AND TABLE_TYPE = 'BASE TABLE'
ORDER BY
    TABLE_NAME;
