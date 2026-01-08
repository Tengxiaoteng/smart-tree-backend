# 知识树节点消失问题 - 解决方案

## 问题根本原因

前端应用使用 **Zustand + persist 中间件**将所有数据存储在**浏览器的 localStorage** 中（存储键名为 `'smart-tree-storage'`），而不是使用后端数据库。

这导致：
1. ✅ 数据只存在于浏览器本地
2. ❌ 清除浏览器数据后，所有节点丢失
3. ❌ 换浏览器或设备后，看不到之前创建的节点
4. ❌ 多设备无法同步数据

虽然前端有 Prisma API 路由（`/api/nodes`），但 Zustand store 并没有调用这些 API 来持久化数据。

## 解决方案

### 方案 1：修改前端，使用后端 API 持久化数据（推荐）

这是最彻底的解决方案，需要修改前端代码。

#### 步骤 1：修改 Zustand store，添加 API 调用

修改 `src/store/useStore.ts`，在添加/更新/删除节点时调用后端 API：

```typescript
// 添加节点
addNode: async (node: KnowledgeNode) => {
  // 1. 先调用后端 API 保存到数据库
  try {
    const response = await fetch('/api/nodes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(node),
    });
    
    if (!response.ok) {
      throw new Error('保存节点失败');
    }
    
    const { data } = await response.json();
    
    // 2. 成功后更新本地状态
    set((state) => {
      const newNodes = { ...state.nodes, [data.id]: data };
      const newRootNodeIds = node.parentId
        ? state.rootNodeIds
        : [...state.rootNodeIds, data.id];
      
      return { nodes: newNodes, rootNodeIds: newRootNodeIds };
    });
  } catch (error) {
    console.error('添加节点失败:', error);
    // 可以显示错误提示
    throw error;
  }
},

// 更新节点
updateNode: async (id: string, updates: Partial<KnowledgeNode>) => {
  try {
    const response = await fetch(`/api/nodes/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    
    if (!response.ok) {
      throw new Error('更新节点失败');
    }
    
    const { data } = await response.json();
    
    set((state) => ({
      nodes: { ...state.nodes, [id]: { ...state.nodes[id], ...data } },
    }));
  } catch (error) {
    console.error('更新节点失败:', error);
    throw error;
  }
},

// 删除节点
deleteNode: async (id: string) => {
  try {
    const response = await fetch(`/api/nodes/${id}`, {
      method: 'DELETE',
    });
    
    if (!response.ok) {
      throw new Error('删除节点失败');
    }
    
    set((state) => {
      const newNodes = { ...state.nodes };
      delete newNodes[id];
      
      return {
        nodes: newNodes,
        rootNodeIds: state.rootNodeIds.filter(nid => nid !== id),
      };
    });
  } catch (error) {
    console.error('删除节点失败:', error);
    throw error;
  }
},
```

#### 步骤 2：添加数据加载函数

在 store 中添加从后端加载数据的函数：

```typescript
interface AppState {
  // ... 其他字段
  
  // 从后端加载数据
  loadNodesFromServer: () => Promise<void>;
  loadTopicsFromServer: () => Promise<void>;
}

// 实现
loadNodesFromServer: async () => {
  try {
    const topicId = get().currentTopicId;
    const url = topicId ? `/api/nodes?topicId=${topicId}` : '/api/nodes';
    
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('加载节点失败');
    }
    
    const { data } = await response.json();
    
    // 转换为 Record 格式
    const nodes: Record<string, KnowledgeNode> = {};
    const rootNodeIds: string[] = [];
    
    data.forEach((node: KnowledgeNode) => {
      nodes[node.id] = node;
      if (!node.parentId) {
        rootNodeIds.push(node.id);
      }
    });
    
    set({ nodes, rootNodeIds });
  } catch (error) {
    console.error('加载节点失败:', error);
  }
},
```

#### 步骤 3：在登录后加载数据

修改 `src/app/page.tsx`，在组件挂载时加载数据：

```typescript
useEffect(() => {
  // 加载数据
  const loadData = async () => {
    await loadTopicsFromServer();
    await loadNodesFromServer();
  };
  
  loadData();
}, []);
```

### 方案 2：临时解决方案 - 不清除 localStorage

如果暂时不想修改代码，可以确保退出登录时不清除 localStorage：

修改 `src/components/Auth/UserMenu.tsx`：

```typescript
const handleLogout = async () => {
  await fetch('/api/auth/logout', { method: 'POST' });
  // 不要清除 localStorage
  // localStorage.clear(); // ← 删除这行（如果有的话）
  router.push('/login');
  router.refresh();
};
```

**缺点：**
- 数据仍然只存在于浏览器本地
- 换浏览器或设备后看不到数据
- 清除浏览器数据后数据丢失

### 方案 3：导出/导入功能

添加数据导出和导入功能，让用户可以手动备份数据：

```typescript
// 导出数据
const exportData = () => {
  const data = localStorage.getItem('smart-tree-storage');
  if (data) {
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `smart-tree-backup-${new Date().toISOString()}.json`;
    a.click();
  }
};

// 导入数据
const importData = (file: File) => {
  const reader = new FileReader();
  reader.onload = (e) => {
    const data = e.target?.result as string;
    localStorage.setItem('smart-tree-storage', data);
    window.location.reload();
  };
  reader.readAsText(file);
};
```

## 推荐实施步骤

1. **立即实施方案 2**：防止退出登录时清除数据（临时解决）
2. **逐步实施方案 1**：
   - 先实现数据加载功能
   - 再实现数据保存功能
   - 最后实现数据同步功能
3. **可选实施方案 3**：作为备份手段

## 注意事项

1. **数据迁移**：实施方案 1 时，需要将现有 localStorage 中的数据迁移到数据库
2. **冲突处理**：如果用户在多个设备上操作，需要处理数据冲突
3. **离线支持**：可以保留 localStorage 作为缓存，实现离线编辑功能
4. **性能优化**：频繁的 API 调用可能影响性能，可以考虑批量操作或防抖

## 测试建议

修改后，测试以下场景：
1. 创建节点 → 退出登录 → 重新登录 → 验证节点是否存在
2. 创建节点 → 清除浏览器数据 → 重新登录 → 验证节点是否存在
3. 在设备 A 创建节点 → 在设备 B 登录 → 验证节点是否同步

