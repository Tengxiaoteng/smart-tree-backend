# 知识树节点消失问题排查指南

## 问题描述
创建知识树后，退出账号再登录，节点就消失了。

## 后端测试结果
✅ **后端数据持久化正常** - 已通过测试验证
✅ **API接口正常** - 创建、查询、登录登出流程完全正常
✅ **用户ID一致性正常** - 退出登录后重新登录，用户ID保持一致

## 问题定位
根据测试结果，**问题出在前端**，而非后端。

## 前端排查步骤

### 1. 检查节点创建时是否传递了 topicId

**问题现象：**
- 如果创建节点时没有传递 `topicId`，节点会被创建，但不会关联到任何主题
- 当前端按 `topicId` 查询时，这些节点不会被返回

**排查方法：**
打开浏览器开发者工具（F12），查看网络请求：

```javascript
// 创建节点的请求应该包含 topicId
POST /api/nodes
{
  "name": "节点名称",
  "topicId": "xxx-xxx-xxx",  // ← 检查这个字段是否存在且正确
  "description": "...",
  ...
}
```

**解决方案：**
确保前端在创建节点时正确传递 `topicId`：
```javascript
const createNode = async (nodeData) => {
  const response = await fetch('/api/nodes', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({
      ...nodeData,
      topicId: currentTopicId  // ← 确保传递了 topicId
    })
  });
};
```

### 2. 检查查询节点时的过滤条件

**问题现象：**
- 查询时使用了错误的 `topicId`
- 查询时没有传递 `topicId`，但期望只看到某个主题的节点

**排查方法：**
查看网络请求：

```javascript
// 查询节点的请求
GET /api/nodes?topicId=xxx-xxx-xxx  // ← 检查 topicId 是否正确
```

**解决方案：**
```javascript
const fetchNodes = async (topicId) => {
  const response = await fetch(`/api/nodes?topicId=${topicId}`, {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
};
```

### 3. 检查 Token 管理

**问题现象：**
- 退出登录后，token 没有被清除
- 重新登录后，使用了旧的 token
- Token 过期但前端没有处理

**排查方法：**
```javascript
// 检查 localStorage 或 sessionStorage
console.log('Token:', localStorage.getItem('token'));
console.log('User ID:', localStorage.getItem('userId'));
```

**解决方案：**
```javascript
// 退出登录时清除所有状态
const logout = () => {
  localStorage.removeItem('token');
  localStorage.removeItem('userId');
  localStorage.removeItem('currentTopicId');
  // 清除其他相关状态
};

// 登录时保存新的 token
const login = async (username, password) => {
  const response = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password })
  });
  const data = await response.json();
  
  localStorage.setItem('token', data.token);
  localStorage.setItem('userId', data.id);
};
```

### 4. 检查前端状态管理

**问题现象：**
- 前端使用了本地状态（如 React state, Vuex, Redux）
- 退出登录后状态没有被清除
- 重新登录后显示的是旧状态

**解决方案：**
```javascript
// React 示例
const logout = () => {
  // 清除 localStorage
  localStorage.clear();
  
  // 重置所有状态
  setNodes([]);
  setTopics([]);
  setCurrentUser(null);
  
  // 跳转到登录页
  navigate('/login');
};
```

## 调试技巧

### 1. 启用后端调试日志
后端已添加调试日志，查看服务器控制台输出：

```
[DEBUG] 用户登录请求 - 用户名: xxx
[DEBUG] 登录成功 - 用户ID: xxx, 用户名: xxx
[DEBUG] 创建节点 - 用户ID: xxx, 节点名称: xxx, topicId: xxx
[DEBUG] 节点创建成功 - ID: xxx, userId: xxx, topicId: xxx
[DEBUG] 获取节点列表 - 用户ID: xxx, topicId: xxx
[DEBUG] 找到 X 个节点
```

### 2. 使用浏览器开发者工具
1. 打开 F12 开发者工具
2. 切换到 Network（网络）标签
3. 执行创建节点、退出登录、重新登录、查询节点的操作
4. 检查每个请求的：
   - 请求 URL
   - 请求头（特别是 Authorization）
   - 请求体（POST 请求）
   - 响应数据

### 3. 直接测试 API
使用提供的测试脚本：

```bash
# 测试数据库持久化
python test_db_persistence.py

# 测试完整 API 流程
python test_api_flow.py
```

## 常见问题

### Q: 节点创建成功，但查询时返回空数组
**A:** 检查查询时的 `topicId` 是否与创建时的 `topicId` 一致

### Q: 重新登录后用户ID变了
**A:** 这不应该发生。如果发生了，说明登录了不同的账号

### Q: Token 过期导致查询失败
**A:** 检查 `ACCESS_TOKEN_EXPIRE_MINUTES` 配置，默认是 30 分钟

## 下一步

1. 按照上述步骤排查前端代码
2. 查看浏览器开发者工具的网络请求
3. 查看后端服务器的调试日志
4. 如果问题依然存在，提供以下信息：
   - 创建节点时的请求和响应
   - 查询节点时的请求和响应
   - 登录时的请求和响应
   - 后端服务器的调试日志

