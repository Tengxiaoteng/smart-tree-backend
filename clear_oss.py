#!/usr/bin/env python3
"""
清空阿里云OSS存储桶中的所有文件
用于生产环境上线前的文件清理
"""

import oss2
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def clear_oss_bucket():
    """清空OSS存储桶中的所有文件"""

    # 从环境变量获取OSS配置
    access_key_id = os.getenv('OSS_ACCESS_KEY_ID')
    access_key_secret = os.getenv('OSS_ACCESS_KEY_SECRET')
    bucket_name = os.getenv('OSS_BUCKET_NAME')
    endpoint = os.getenv('OSS_ENDPOINT')

    if not all([access_key_id, access_key_secret, bucket_name, endpoint]):
        print("错误: OSS配置信息不完整")
        return

    print(f"OSS配置:")
    print(f"  Bucket: {bucket_name}")
    print(f"  Endpoint: {endpoint}")

    # 确认操作
    print("\n⚠️  警告: 此操作将删除OSS存储桶中的所有文件！")
    print("⚠️  这是不可逆的操作！")
    confirm = input("\n请输入 'YES' 确认继续: ")

    if confirm != 'YES':
        print("操作已取消")
        return

    try:
        # 创建Bucket对象
        auth = oss2.Auth(access_key_id, access_key_secret)
        bucket = oss2.Bucket(auth, endpoint, bucket_name)

        print("\n开始扫描文件...")

        # 列出所有文件
        all_objects = []
        for obj in oss2.ObjectIterator(bucket):
            all_objects.append(obj.key)

        if not all_objects:
            print("✓ OSS存储桶中没有文件")
            return

        print(f"\n找到 {len(all_objects)} 个文件")

        # 显示前10个文件作为示例
        print("\n文件示例（前10个）:")
        for i, key in enumerate(all_objects[:10]):
            print(f"  {i+1}. {key}")
        if len(all_objects) > 10:
            print(f"  ... 还有 {len(all_objects) - 10} 个文件")

        # 再次确认
        confirm2 = input(f"\n确认删除这 {len(all_objects)} 个文件？输入 'DELETE' 继续: ")
        if confirm2 != 'DELETE':
            print("操作已取消")
            return

        print("\n开始删除文件...")

        # 批量删除（OSS支持每次最多删除1000个对象）
        batch_size = 1000
        deleted_count = 0

        for i in range(0, len(all_objects), batch_size):
            batch = all_objects[i:i+batch_size]
            try:
                result = bucket.batch_delete_objects(batch)
                deleted_count += len(batch)
                print(f"✓ 已删除 {deleted_count}/{len(all_objects)} 个文件")

                # 检查是否有删除失败的
                if hasattr(result, 'deleted_keys'):
                    failed = set(batch) - set(result.deleted_keys)
                    if failed:
                        print(f"⚠️  以下文件删除失败: {failed}")
            except Exception as e:
                print(f"✗ 批量删除失败: {e}")

        print(f"\n✓ OSS清理完成！共删除 {deleted_count} 个文件")

        # 验证
        remaining = sum(1 for _ in oss2.ObjectIterator(bucket))
        if remaining == 0:
            print("✓ OSS存储桶已完全清空")
        else:
            print(f"⚠️  还有 {remaining} 个文件未删除")

    except oss2.exceptions.OssError as e:
        print(f"\n✗ OSS错误: {e}")
    except Exception as e:
        print(f"\n✗ 错误: {e}")

if __name__ == "__main__":
    clear_oss_bucket()
