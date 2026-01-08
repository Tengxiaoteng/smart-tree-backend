# FastAPI 后端集成指南

## 问题解决方案

已成功将前端从 localStorage 改为使用 FastAPI 后端进行数据持久化！

## 修改内容

### 1. 创建 API 客户端 (`src/lib/api.ts`)

新建了 FastAPI 客户端工具类，包含：
- **认证 API**: `authAPI.login()`, `authAPI.register()`, `authAPI.logout()`
- **主题 API**: `topicAPI.getAll()`, `topicAPI.create()`, `topicAPI.update()`, `topicAPI.delete()`
- **节点 API**: `nodeAPI.getAll()`, `nodeAPI.create()`, `nodeAPI.update()`, `nodeAPI.delete()`

Token 存储在 `localStorage` 的 `fastapi_token` 键中。

### 2. 修改认证页面

**登录页面** (`src/app/login/page.tsx`):
```typescript
import { authAPI } from '@/lib/api';

// 使用 FastAPI 登录
await authAPI.login({ username, password });
```

**注册页面** (`src/app/register/page.tsx`):
```typescript
import { authAPI } from '@/lib/api';

// 使用 FastAPI 注册
await authAPI.register({ username, password, nickname });
```

**用户菜单** (`src/components/Auth/UserMenu.tsx`):
```typescript
import { authAPI } from '@/lib/api';

// 登出时只清除 token，不清除应用数据
authAPI.logout();
```

### 3. 修改 Zustand Store (`src/store/useStore.ts`)

添加了与 FastAPI 后端的数据同步：

**创建节点**:
```typescript
addNode: (node) => {
  // 1. 先更新本地状态（立即响应）
  set((state) => { ... });
  
  // 2. 异步同步到 FastAPI 后端
  nodeAPI.create(node).catch(console.error);
}
```

**更新节点**:
```typescript
updateNode: (id, updates) => {
  set((state) => { ... });
  nodeAPI.update(id, updates).catch(console.error);
}
```

**删除节点**:
```typescript
deleteNode: (id) => {
  set((state) => { ... });
  nodeAPI.delete(id).catch(console.error);
}
```

**从服务器加载数据**:
```typescript
loadTopicsFromServer: async () => {
  const topics = await topicAPI.getAll();
  set({ topics: topicsRecord });
}

loadNodesFromServer: async () => {
  const nodes = await nodeAPI.getAll(topicId);
  set({ nodes: nodesRecord, rootNodeIds });
}
```

### 4. 修改主页面 (`src/app/page.tsx`)

在组件挂载时从 FastAPI 后端加载数据：

```typescript
useEffect(() => {
  const loadData = async () => {
    await loadTopicsFromServer();
    await loadNodesFromServer();
  };
  
  loadData();
}, [loadTopicsFromServer, loadNodesFromServer]);
```

### 5. 环境变量配置 (`.env`)

添加了 FastAPI 后端 URL：
```
NEXT_PUBLIC_API_URL="http://localhost:8000"
```

## 数据流程

### 旧流程（有问题）
```
用户操作 → Zustand Store → localStorage
                              ↓
                         ❌ 清除浏览器数据后丢失
```

### 新流程（已修复）
```
用户操作 → Zustand Store → localStorage (缓存)
                         ↓
                    FastAPI 后端 → MySQL 数据库
                         ↓
                    ✅ 数据持久化
```

## 启动步骤

### 1. 启动 FastAPI 后端

```bash
cd /Users/junteng_dong/Desktop/smart-tree-backend

# 激活虚拟环境（如果有）
source venv/bin/activate

# 启动服务器
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

后端应该运行在 `http://localhost:8000`

### 2. 启动前端

```bash
cd ../postgraduate/收集/smart-tree

# 安装依赖（首次运行）
npm install

# 启动开发服务器
npm run dev
```

前端应该运行在 `http://localhost:3000`

## 测试验证

### 方法 1: 使用测试脚本

```bash
cd /Users/junteng_dong/Desktop/smart-tree-backend
python test_frontend_backend_integration.py
```

这个脚本会自动测试：
1. 注册用户
2. 创建主题
3. 创建节点
4. 退出登录
5. 重新登录
6. 验证节点仍然存在

### 方法 2: 手动测试

1. 打开浏览器访问 `http://localhost:3000`
2. 注册/登录账号
3. 创建一个主题
4. 创建一些知识节点
5. 点击退出登录
6. 重新登录
7. **验证**: 之前创建的节点应该还在！

### 方法 3: 跨设备测试

1. 在设备 A 上登录并创建节点
2. 在设备 B 上登录同一账号
3. **验证**: 应该能看到设备 A 创建的节点

## 技术细节

### 认证机制

- **后端**: FastAPI 使用 JWT Token 认证
- **前端**: Token 存储在 `localStorage.fastapi_token`
- **请求头**: `Authorization: Bearer <token>`

### 数据同步策略

采用**乐观更新**策略：
1. 先更新本地状态（立即响应，用户体验好）
2. 异步同步到后端（不阻塞 UI）
3. 如果同步失败，在控制台输出错误（可以后续添加重试机制）

### localStorage 的作用

现在 localStorage 作为**缓存**使用：
- 提供离线查看功能
- 加快页面加载速度
- 减少不必要的 API 请求

但数据的**真实来源**是 FastAPI 后端的 MySQL 数据库。

## 注意事项

1. **确保后端运行**: 前端依赖 FastAPI 后端，必须先启动后端
2. **CORS 配置**: 后端已配置允许前端跨域请求
3. **Token 过期**: JWT Token 有效期为 7 天，过期后需要重新登录
4. **网络错误**: 如果 API 调用失败，会在控制台输出错误信息

## 下一步优化建议

1. **错误处理**: 添加用户友好的错误提示
2. **重试机制**: API 失败时自动重试
3. **离线支持**: 检测网络状态，离线时使用缓存
4. **数据同步**: 实现冲突解决机制
5. **加载状态**: 显示数据加载进度

## 问题排查

### 前端无法连接后端

检查：
- FastAPI 后端是否运行在 `http://localhost:8000`
- 浏览器控制台是否有 CORS 错误
- `.env` 文件中的 `NEXT_PUBLIC_API_URL` 是否正确

### 登录后看不到数据

检查：
- 浏览器控制台的 Network 标签，查看 API 请求是否成功
- 检查 `localStorage.fastapi_token` 是否存在
- 后端日志是否有错误信息

### 节点创建后消失

检查：
- 浏览器控制台是否有 API 错误
- 后端数据库中是否真的保存了数据
- 刷新页面后是否能看到节点（验证加载功能）

## 总结

✅ 问题已解决！现在数据会真正保存到 FastAPI 后端的 MySQL 数据库中。

✅ 退出登录后重新登录，节点不会消失。

✅ 支持多设备同步，在任何设备上登录都能看到数据。

