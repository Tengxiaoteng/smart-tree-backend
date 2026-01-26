#!/bin/bash

# 智学树应用启动脚本
# 用于同时启动 FastAPI 后端和 Next.js 前端

echo "=========================================="
echo "  智学树 (Smart Tree) 应用启动脚本"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查后端是否已经在运行
echo -e "${BLUE}[1/4] 检查后端状态...${NC}"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ 后端已在运行 (http://localhost:8000)${NC}"
    BACKEND_RUNNING=true
else
    echo -e "${RED}✗ 后端未运行${NC}"
    BACKEND_RUNNING=false
fi

# 如果后端未运行，启动后端
if [ "$BACKEND_RUNNING" = false ]; then
    echo ""
    echo -e "${BLUE}[2/4] 启动 FastAPI 后端...${NC}"
    
    # 检查是否在正确的目录
    if [ ! -f "main.py" ]; then
        echo -e "${RED}错误: 请在 smart-tree-backend 目录下运行此脚本${NC}"
        exit 1
    fi
    
    # 启动后端（在后台运行）
    # 默认优先使用本地 SQLite（避免依赖外部 MySQL/网络）；可通过 ENV_FILE 覆盖。
    ENV_FILE_TO_USE="${ENV_FILE}"
    if [ -z "$ENV_FILE_TO_USE" ]; then
      if [ -f ".env.sqlite" ]; then
        ENV_FILE_TO_USE=".env.sqlite"
      elif [ -f ".env.sqlite.example" ]; then
        ENV_FILE_TO_USE=".env.sqlite.example"
      else
        ENV_FILE_TO_USE=".env"
      fi
    fi

    echo "启动命令: ENV_FILE=${ENV_FILE_TO_USE} uvicorn main:app --reload --host 0.0.0.0 --port 8000"
    nohup env ENV_FILE="${ENV_FILE_TO_USE}" uvicorn main:app --reload --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
    BACKEND_PID=$!
    
    echo "后端进程 PID: $BACKEND_PID"
    echo "日志文件: backend.log"
    
    # 等待后端启动
    echo "等待后端启动..."
    for i in {1..10}; do
        sleep 1
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓ 后端启动成功！${NC}"
            break
        fi
        echo -n "."
    done
    echo ""
else
    echo ""
    echo -e "${BLUE}[2/4] 跳过后端启动（已在运行）${NC}"
fi

# 检查前端目录
echo ""
echo -e "${BLUE}[3/4] 检查前端目录...${NC}"
FRONTEND_DIR="../smart-tree"

if [ ! -d "$FRONTEND_DIR" ]; then
    echo -e "${RED}错误: 前端目录不存在: $FRONTEND_DIR${NC}"
    exit 1
fi

if [ ! -f "$FRONTEND_DIR/package.json" ]; then
    echo -e "${RED}错误: 前端目录中没有 package.json${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 前端目录存在${NC}"

# 启动前端
echo ""
echo -e "${BLUE}[4/4] 启动 Next.js 前端...${NC}"
echo "前端目录: $FRONTEND_DIR"
echo ""
echo -e "${GREEN}提示: 前端将在前台运行，按 Ctrl+C 停止${NC}"
echo ""

cd "$FRONTEND_DIR"

# 检查是否需要安装依赖
if [ ! -d "node_modules" ]; then
    echo "首次运行，安装依赖..."
    npm install
fi

# 启动前端开发服务器
echo ""
echo "=========================================="
echo "  启动前端开发服务器"
echo "=========================================="
echo ""
npm run dev

# 如果前端被停止，询问是否停止后端
echo ""
echo -e "${BLUE}前端已停止${NC}"

if [ "$BACKEND_RUNNING" = false ] && [ ! -z "$BACKEND_PID" ]; then
    read -p "是否停止后端服务器? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "停止后端..."
        kill $BACKEND_PID
        echo -e "${GREEN}✓ 后端已停止${NC}"
    else
        echo -e "${BLUE}后端继续运行 (PID: $BACKEND_PID)${NC}"
        echo "如需停止，运行: kill $BACKEND_PID"
    fi
fi

echo ""
echo "=========================================="
echo "  应用已关闭"
echo "=========================================="
