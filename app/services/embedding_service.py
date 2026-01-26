"""
Embedding 服务 - 使用 DashScope text-embedding-v4

支持功能：
- 文本向量化
- 批量处理
- 向量相似度计算
- 简单向量存储和检索
- 资料智能分块和索引（V3智能出题）
"""

import re
import json
import hashlib
import numpy as np
from typing import Optional
from pathlib import Path
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embedding 服务，使用 DashScope text-embedding-v4"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "text-embedding-v4",
        dimension: int = 512,  # 支持 64-2048，512 是性价比较好的选择
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        """
        将单个文本转换为向量

        Args:
            text: 输入文本（最大 8192 tokens）

        Returns:
            向量列表
        """
        # 截断过长的文本
        if len(text) > 8000:
            text = text[:8000]

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimension,
        )
        return response.data[0].embedding

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        批量将文本转换为向量

        Args:
            texts: 文本列表（每批最多 10 个）

        Returns:
            向量列表
        """
        if not texts:
            return []

        # 截断过长的文本
        texts = [t[:8000] if len(t) > 8000 else t for t in texts]

        # 分批处理（每批最多 10 个）
        all_embeddings = []
        batch_size = 10

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimension,
            )
            # 按 index 排序确保顺序正确
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([d.embedding for d in sorted_data])

        return all_embeddings

    @staticmethod
    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """计算两个向量的余弦相似度"""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    @staticmethod
    def batch_cosine_similarity(query_vec: list[float], vectors: list[list[float]]) -> list[float]:
        """批量计算余弦相似度"""
        if not vectors:
            return []
        query = np.array(query_vec)
        matrix = np.array(vectors)
        # 归一化
        query_norm = query / np.linalg.norm(query)
        matrix_norms = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        # 计算相似度
        similarities = np.dot(matrix_norms, query_norm)
        return similarities.tolist()

    # ===== V3智能出题：资料分块和索引 =====

    def smart_split(self, text: str, target_size: int = 400, overlap: int = 50) -> list[str]:
        """
        智能分块：优先按段落分割，保持语义完整性

        Args:
            text: 输入文本
            target_size: 目标分块大小（字符数）
            overlap: 块之间的重叠字符数

        Returns:
            分块列表
        """
        if not text or len(text) <= target_size:
            return [text] if text else []

        # 1. 首先按双换行分割段落
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # 如果当前段落本身就超过目标大小，需要进一步分割
            if len(para) > target_size:
                # 先保存当前累积的内容
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # 按句子分割长段落
                sentences = re.split(r'([。！？；\n])', para)
                temp_chunk = ""

                for i in range(0, len(sentences) - 1, 2):
                    sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
                    if len(temp_chunk) + len(sentence) <= target_size:
                        temp_chunk += sentence
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sentence

                if temp_chunk:
                    current_chunk = temp_chunk

            elif len(current_chunk) + len(para) + 2 <= target_size:
                # 可以添加到当前块
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                # 当前块已满，开始新块
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para

        # 保存最后一块
        if current_chunk:
            chunks.append(current_chunk.strip())

        # 过滤空块
        chunks = [c for c in chunks if c and len(c) > 20]

        return chunks

    def index_material_chunks(
        self,
        material_id: str,
        content: str,
        chunk_size: int = 400,
    ) -> list[dict]:
        """
        将资料分块并生成向量索引

        Args:
            material_id: 资料ID
            content: 资料内容
            chunk_size: 分块大小

        Returns:
            分块列表，每项包含 {chunk_id, material_id, text, embedding, chunk_index}
        """
        # 1. 智能分块
        chunks = self.smart_split(content, chunk_size)

        if not chunks:
            return []

        # 2. 批量向量化
        try:
            embeddings = self.embed_texts(chunks)
        except Exception as e:
            logger.error(f"向量化失败: {e}")
            return []

        # 3. 构建索引
        indexed_chunks = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            indexed_chunks.append({
                "chunk_id": f"{material_id}_chunk_{idx}",
                "material_id": material_id,
                "text": chunk,
                "embedding": embedding,
                "chunk_index": idx,
                "token_count": len(chunk),  # 简单估算
            })

        logger.info(f"资料 {material_id} 分块完成: {len(indexed_chunks)} 个分块")
        return indexed_chunks

    def retrieve_relevant_chunks(
        self,
        query_embedding: list[float],
        chunks: list[dict],
        top_k: int = 5,
        min_similarity: float = 0.4,
    ) -> list[dict]:
        """
        从分块中检索相关内容

        Args:
            query_embedding: 查询向量
            chunks: 分块列表，每项需包含 embedding 字段
            top_k: 返回前K个结果
            min_similarity: 最小相似度阈值

        Returns:
            检索结果列表，按相似度降序排列
        """
        if not chunks:
            return []

        # 提取所有向量
        embeddings = [c.get("embedding", []) for c in chunks if c.get("embedding")]

        if not embeddings:
            return []

        # 批量计算相似度
        similarities = self.batch_cosine_similarity(query_embedding, embeddings)

        # 组合结果并过滤
        results = []
        for i, chunk in enumerate(chunks):
            if not chunk.get("embedding"):
                continue
            sim = similarities[i] if i < len(similarities) else 0
            if sim >= min_similarity:
                results.append({
                    **chunk,
                    "similarity": sim,
                })

        # 按相似度排序
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results[:top_k]


class SimpleVectorStore:
    """
    简单的向量存储

    用于存储和检索节点/树的 Embedding
    生产环境建议使用 Milvus / Qdrant / pgvector
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Args:
            storage_path: 持久化存储路径（JSON 文件），None 表示仅内存存储
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.vectors: dict[str, dict] = {}  # id -> {embedding, metadata, text}
        self._load()

    def _load(self):
        """从文件加载"""
        if self.storage_path and self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.vectors = json.load(f)
                logger.info(f"从 {self.storage_path} 加载了 {len(self.vectors)} 个向量")
            except Exception as e:
                logger.warning(f"加载向量存储失败: {e}")
                self.vectors = {}

    def _save(self):
        """保存到文件"""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f, ensure_ascii=False)

    def add(
        self,
        id: str,
        embedding: list[float],
        text: str = "",
        metadata: Optional[dict] = None,
    ):
        """
        添加或更新一个向量

        Args:
            id: 唯一标识符（如 node_id 或 tree_id）
            embedding: 向量
            text: 原始文本（用于调试）
            metadata: 元数据（如 name, type 等）
        """
        self.vectors[id] = {
            "embedding": embedding,
            "text": text[:500] if text else "",  # 只存前 500 字
            "metadata": metadata or {},
        }
        self._save()

    def add_batch(
        self,
        items: list[dict],  # [{id, embedding, text, metadata}, ...]
    ):
        """批量添加向量"""
        for item in items:
            self.vectors[item["id"]] = {
                "embedding": item["embedding"],
                "text": item.get("text", "")[:500],
                "metadata": item.get("metadata", {}),
            }
        self._save()

    def get(self, id: str) -> Optional[dict]:
        """获取单个向量"""
        return self.vectors.get(id)

    def delete(self, id: str):
        """删除向量"""
        if id in self.vectors:
            del self.vectors[id]
            self._save()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_fn: Optional[callable] = None,
        min_similarity: float = 0.0,
    ) -> list[dict]:
        """
        向量检索

        Args:
            query_embedding: 查询向量
            top_k: 返回前 K 个结果
            filter_fn: 过滤函数，接收 (id, metadata)，返回 True 保留
            min_similarity: 最小相似度阈值

        Returns:
            结果列表，每项包含 {id, similarity, text, metadata}
        """
        if not self.vectors:
            return []

        # 过滤
        candidates = []
        for id, data in self.vectors.items():
            if filter_fn and not filter_fn(id, data.get("metadata", {})):
                continue
            candidates.append((id, data))

        if not candidates:
            return []

        # 计算相似度
        ids = [c[0] for c in candidates]
        embeddings = [c[1]["embedding"] for c in candidates]
        similarities = EmbeddingService.batch_cosine_similarity(query_embedding, embeddings)

        # 组合结果
        results = []
        for i, (id, data) in enumerate(candidates):
            sim = similarities[i]
            if sim >= min_similarity:
                results.append({
                    "id": id,
                    "similarity": sim,
                    "text": data.get("text", ""),
                    "metadata": data.get("metadata", {}),
                })

        # 排序并取 top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def search_by_type(
        self,
        query_embedding: list[float],
        type_value: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> list[dict]:
        """
        按类型检索（如只检索树或只检索节点）

        Args:
            query_embedding: 查询向量
            type_value: metadata 中的 type 值（如 "tree" 或 "node"）
            top_k: 返回前 K 个结果
            min_similarity: 最小相似度阈值
        """
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_fn=lambda id, meta: meta.get("type") == type_value,
            min_similarity=min_similarity,
        )

    def get_all_by_type(self, type_value: str) -> list[dict]:
        """获取指定类型的所有向量"""
        results = []
        for id, data in self.vectors.items():
            if data.get("metadata", {}).get("type") == type_value:
                results.append({
                    "id": id,
                    "text": data.get("text", ""),
                    "metadata": data.get("metadata", {}),
                })
        return results

    def clear(self):
        """清空所有向量"""
        self.vectors = {}
        self._save()

    def __len__(self):
        return len(self.vectors)


def build_node_text(node) -> str:
    """
    将节点信息组合成适合 Embedding 的文本

    Args:
        node: MockNode 或类似对象

    Returns:
        组合后的文本
    """
    parts = [node.name]
    if hasattr(node, 'description') and node.description:
        parts.append(node.description)
    if hasattr(node, 'key_concepts') and node.key_concepts:
        parts.append("关键概念: " + ", ".join(node.key_concepts))
    return " | ".join(parts)


def build_tree_text(tree, root_nodes: list = None) -> str:
    """
    将树信息组合成适合 Embedding 的文本

    Args:
        tree: MockTree 或类似对象
        root_nodes: 根节点列表（可选）

    Returns:
        组合后的文本
    """
    parts = [tree.name]
    if hasattr(tree, 'scope') and tree.scope:
        parts.append("范围: " + ", ".join(tree.scope))
    if hasattr(tree, 'keywords') and tree.keywords:
        parts.append("关键词: " + ", ".join(tree.keywords))
    if root_nodes:
        root_names = [n.name for n in root_nodes[:5]]
        parts.append("主要节点: " + ", ".join(root_names))
    return " | ".join(parts)
