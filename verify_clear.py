#!/usr/bin/env python3
import pymysql
from dotenv import load_dotenv
import os
import re

load_dotenv()
url = os.getenv('DATABASE_URL')
pattern = r'mysql\+pymysql://([^:]+):([^@]+)@([^:]+):(\d+)/([^?]+)'
match = re.match(pattern, url)
username, password, host, port, database = match.groups()
password = password.replace('%40', '@')

conn = pymysql.connect(host=host, port=int(port), user=username, password=password, database=database, charset='utf8mb4')
cursor = conn.cursor()

cursor.execute('SHOW TABLES')
tables = cursor.fetchall()

print(f'数据库: {host}/{database}')
print(f'验证清空结果:\n')

total_rows = 0
for table_tuple in tables:
    table_name = table_tuple[0]
    cursor.execute(f'SELECT COUNT(*) FROM `{table_name}`')
    count = cursor.fetchone()[0]
    total_rows += count
    if count > 0:
        print(f'⚠️  {table_name}: {count} 行 (未清空)')
    else:
        print(f'✓ {table_name}: 0 行')

print(f'\n总数据行数: {total_rows}')
if total_rows == 0:
    print('✓ 所有表已完全清空！')
else:
    print(f'⚠️  还有 {total_rows} 行数据未清空')

cursor.close()
conn.close()
