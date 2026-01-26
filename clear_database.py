#!/usr/bin/env python3
"""
清空数据库所有表的数据，但保留表结构
用于生产环境上线前的数据清理
"""

import pymysql
import os
from dotenv import load_dotenv
import re

# 加载环境变量
load_dotenv()

def parse_database_url(url):
    """解析数据库连接URL"""
    # mysql+pymysql://user:password@host:port/database?charset=utf8mb4
    pattern = r'mysql\+pymysql://([^:]+):([^@]+)@([^:]+):(\d+)/([^?]+)'
    match = re.match(pattern, url)
    if not match:
        raise ValueError(f"无法解析数据库URL: {url}")

    username, password, host, port, database = match.groups()
    # URL解码密码（处理特殊字符）
    password = password.replace('%40', '@')

    return {
        'host': host,
        'port': int(port),
        'user': username,
        'password': password,
        'database': database,
        'charset': 'utf8mb4'
    }

def clear_all_tables():
    """清空所有表的数据"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("错误: 未找到 DATABASE_URL 环境变量")
        return

    # 解析数据库连接信息
    db_config = parse_database_url(database_url)

    print(f"连接到数据库: {db_config['host']}:{db_config['port']}/{db_config['database']}")
    print(f"用户: {db_config['user']}")

    # 确认操作
    print("\n⚠️  警告: 此操作将删除所有表中的数据，但保留表结构！")
    print("⚠️  这是不可逆的操作！")
    confirm = input("\n请输入 'YES' 确认继续: ")

    if confirm != 'YES':
        print("操作已取消")
        return

    try:
        # 连接数据库
        connection = pymysql.connect(**db_config)
        cursor = connection.cursor()

        # 禁用外键检查
        print("\n禁用外键检查...")
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")

        # 获取所有表名
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()

        if not tables:
            print("数据库中没有找到任何表")
            return

        print(f"\n找到 {len(tables)} 个表:")
        for table in tables:
            print(f"  - {table[0]}")

        # 清空每个表
        print("\n开始清空表数据...")
        for table in tables:
            table_name = table[0]
            try:
                cursor.execute(f"TRUNCATE TABLE `{table_name}`;")
                print(f"✓ 已清空表: {table_name}")
            except Exception as e:
                print(f"✗ 清空表 {table_name} 失败: {e}")

        # 重新启用外键检查
        print("\n重新启用外键检查...")
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")

        # 提交更改
        connection.commit()

        print("\n✓ 所有表数据已清空完成！")
        print("✓ 表结构已保留")

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        if 'connection' in locals():
            connection.rollback()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()
            print("\n数据库连接已关闭")

if __name__ == "__main__":
    clear_all_tables()
