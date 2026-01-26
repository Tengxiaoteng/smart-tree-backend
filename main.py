import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError
from app.core.config import settings
from app.core.database import Base, engine
from app.models import User, Topic, KnowledgeNode, UserSettings, Material
from app.routes import topics, nodes, auth, materials, questions, answer_records, node_material_links, node_prerequisites, user_notes, user_settings, user_profile, upload, files, credits, llm, batches, tree, community, health, shares, node_expansion, tree_settings


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    _run_startup()
    yield


# 创建 FastAPI 应用
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

@app.exception_handler(SQLAlchemyError)
async def _sqlalchemy_error_handler(request: Request, exc: SQLAlchemyError):
    # 统一把数据库异常转换成 JSON 响应，避免未处理异常导致浏览器端出现 CORS 级别的 `Failed to fetch`
    logging.exception("数据库异常: %s", exc)
    detail = "数据库错误，请检查数据库连接与表结构（可能需要执行迁移）"
    if os.getenv("DEBUG_DB_ERRORS", "false").lower() == "true":
        detail = f"{detail}: {exc}"
    return JSONResponse(status_code=500, content={"detail": detail})

def _run_startup() -> None:
    """应用启动时执行：创建表 + 运行迁移"""
    auto_create_tables = os.getenv("AUTO_CREATE_TABLES", "true").lower() == "true"
    if auto_create_tables:
        try:
            Base.metadata.create_all(bind=engine)
        except SQLAlchemyError as exc:
            logging.exception("数据库初始化失败（无法创建表），请检查 DATABASE_URL 连接与权限: %s", exc)

    auto_migrate_schema = os.getenv("AUTO_MIGRATE_SCHEMA", "true").lower() == "true"
    if not auto_migrate_schema:
        return

    # 轻量“自迁移”：只补齐 Material 的 AI 字段列，避免新功能上线后旧库缺列导致读取/更新报错。
    try:
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        if "material" in tables:
            existing_cols = {col["name"] for col in inspector.get_columns("material")}
            is_mysql = engine.dialect.name == "mysql"

            alter_statements: list[str] = []

            def _add(stmt_mysql: str, stmt_other: str) -> None:
                alter_statements.append(stmt_mysql if is_mysql else stmt_other)

            if "organizedContent" not in existing_cols:
                _add(
                    "ALTER TABLE material ADD COLUMN organizedContent LONGTEXT NULL",
                    "ALTER TABLE material ADD COLUMN organizedContent TEXT",
                )
            if "aiSummary" not in existing_cols:
                _add(
                    "ALTER TABLE material ADD COLUMN aiSummary LONGTEXT NULL",
                    "ALTER TABLE material ADD COLUMN aiSummary TEXT",
                )
            if "extractedConcepts" not in existing_cols:
                _add(
                    "ALTER TABLE material ADD COLUMN extractedConcepts JSON NULL",
                    "ALTER TABLE material ADD COLUMN extractedConcepts TEXT",
                )
            if "isOrganized" not in existing_cols:
                _add(
                    "ALTER TABLE material ADD COLUMN isOrganized TINYINT(1) NOT NULL DEFAULT 0",
                    "ALTER TABLE material ADD COLUMN isOrganized BOOLEAN NOT NULL DEFAULT 0",
                )
            if "structuredContent" not in existing_cols:
                _add(
                    "ALTER TABLE material ADD COLUMN structuredContent JSON NULL",
                    "ALTER TABLE material ADD COLUMN structuredContent TEXT",
                )
            if "isStructured" not in existing_cols:
                _add(
                    "ALTER TABLE material ADD COLUMN isStructured TINYINT(1) NOT NULL DEFAULT 0",
                    "ALTER TABLE material ADD COLUMN isStructured BOOLEAN NOT NULL DEFAULT 0",
                )

            if alter_statements:
                with engine.begin() as conn:
                    for stmt in alter_statements:
                        conn.execute(text(stmt))

                logging.info("已自动补齐 material 表字段: %s", ", ".join(sorted(set(alter_statements))))
    except SQLAlchemyError as exc:
        logging.exception("数据库自迁移失败（material 表缺少列且无法自动补齐）: %s", exc)
        raise

    # 补齐 user_profile 表字段（例如后续新增的 email）
    try:
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        if "user_profile" in tables:
            existing_cols = {col["name"] for col in inspector.get_columns("user_profile")}
            is_mysql = engine.dialect.name == "mysql"

            alter_statements: list[str] = []

            if "email" not in existing_cols:
                alter_statements.append(
                    "ALTER TABLE user_profile ADD COLUMN email VARCHAR(255) NULL"
                    if is_mysql
                    else "ALTER TABLE user_profile ADD COLUMN email VARCHAR(255)"
                )

            if alter_statements:
                with engine.begin() as conn:
                    for stmt in alter_statements:
                        conn.execute(text(stmt))

                logging.info("已自动补齐 user_profile 表字段: %s", ", ".join(sorted(set(alter_statements))))
    except SQLAlchemyError as exc:
        logging.exception("数据库自迁移失败（user_profile 表缺少列且无法自动补齐）: %s", exc)
        raise

    # 补齐 question 表字段（兼容旧库：缺列会导致 /questions 查询直接 500，从而出现“部分数据加载失败”）
    try:
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        if "question" in tables:
            existing_cols = {col["name"] for col in inspector.get_columns("question")}
            is_mysql = engine.dialect.name == "mysql"

            alter_statements: list[str] = []

            def _add(stmt_mysql: str, stmt_other: str) -> None:
                alter_statements.append(stmt_mysql if is_mysql else stmt_other)

            if "hints" not in existing_cols:
                _add(
                    "ALTER TABLE question ADD COLUMN hints JSON NULL",
                    "ALTER TABLE question ADD COLUMN hints TEXT",
                )
            if "relatedConcepts" not in existing_cols:
                _add(
                    "ALTER TABLE question ADD COLUMN relatedConcepts JSON NULL",
                    "ALTER TABLE question ADD COLUMN relatedConcepts TEXT",
                )
            if "parentQuestionId" not in existing_cols:
                _add(
                    "ALTER TABLE question ADD COLUMN parentQuestionId VARCHAR(191) NULL",
                    "ALTER TABLE question ADD COLUMN parentQuestionId VARCHAR(191)",
                )
            if "derivationType" not in existing_cols:
                _add(
                    "ALTER TABLE question ADD COLUMN derivationType VARCHAR(255) NULL",
                    "ALTER TABLE question ADD COLUMN derivationType VARCHAR(255)",
                )

            if alter_statements:
                with engine.begin() as conn:
                    for stmt in alter_statements:
                        conn.execute(text(stmt))

                    # 兼容旧字段：把 derivedFromQuestionId 迁移到 parentQuestionId，避免“派生题”关系丢失
                    if "derivedFromQuestionId" in existing_cols:
                        conn.execute(
                            text(
                                "UPDATE question "
                                "SET parentQuestionId = derivedFromQuestionId "
                                "WHERE parentQuestionId IS NULL AND derivedFromQuestionId IS NOT NULL"
                            )
                        )

                    # 兼容旧字段：用 isDerivedQuestion 给 derivationType 一个默认值
                    if "isDerivedQuestion" in existing_cols:
                        conn.execute(
                            text(
                                "UPDATE question "
                                "SET derivationType = 'derived' "
                                "WHERE derivationType IS NULL AND isDerivedQuestion = 1"
                            )
                        )

                logging.info("已自动补齐 question 表字段: %s", ", ".join(sorted(set(alter_statements))))
    except SQLAlchemyError as exc:
        logging.exception("数据库自迁移失败（question 表缺少列且无法自动补齐）: %s", exc)
        raise

    # 补齐 answerrecord 表字段（追问对话持久化）
    try:
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        if "answerrecord" in tables:
            existing_cols = {col["name"] for col in inspector.get_columns("answerrecord")}
            is_mysql = engine.dialect.name == "mysql"

            alter_statements: list[str] = []

            if "followUpMessages" not in existing_cols:
                alter_statements.append(
                    "ALTER TABLE answerrecord ADD COLUMN followUpMessages JSON NULL"
                    if is_mysql
                    else "ALTER TABLE answerrecord ADD COLUMN followUpMessages TEXT"
                )

            if alter_statements:
                with engine.begin() as conn:
                    for stmt in alter_statements:
                        conn.execute(text(stmt))

                logging.info("已自动补齐 answerrecord 表字段: %s", ", ".join(sorted(set(alter_statements))))
    except SQLAlchemyError as exc:
        logging.exception("数据库自迁移失败（answerrecord 表缺少列且无法自动补齐）: %s", exc)
        raise

    # 补齐 knowledgenode 表字段（学习状态跟踪）
    try:
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        if "knowledgenode" in tables:
            existing_cols = {col["name"] for col in inspector.get_columns("knowledgenode")}
            is_mysql = engine.dialect.name == "mysql"

            alter_statements: list[str] = []

            if "learningStatus" not in existing_cols:
                alter_statements.append(
                    "ALTER TABLE knowledgenode ADD COLUMN learningStatus JSON NULL"
                    if is_mysql
                    else "ALTER TABLE knowledgenode ADD COLUMN learningStatus TEXT"
                )

            if alter_statements:
                with engine.begin() as conn:
                    for stmt in alter_statements:
                        conn.execute(text(stmt))

                logging.info("已自动补齐 knowledgenode 表字段: %s", ", ".join(sorted(set(alter_statements))))
    except SQLAlchemyError as exc:
        logging.exception("数据库自迁移失败（knowledgenode 表缺少列且无法自动补齐）: %s", exc)
        raise

    # 补齐 material 表的快速匹配字段
    try:
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        if "material" in tables:
            existing_cols = {col["name"] for col in inspector.get_columns("material")}
            is_mysql = engine.dialect.name == "mysql"

            alter_statements: list[str] = []

            if "contentDigest" not in existing_cols:
                alter_statements.append(
                    "ALTER TABLE material ADD COLUMN contentDigest VARCHAR(255) NULL"
                    if is_mysql
                    else "ALTER TABLE material ADD COLUMN contentDigest VARCHAR(255)"
                )
            if "keyTopics" not in existing_cols:
                alter_statements.append(
                    "ALTER TABLE material ADD COLUMN keyTopics JSON NULL"
                    if is_mysql
                    else "ALTER TABLE material ADD COLUMN keyTopics TEXT"
                )
            if "contentHash" not in existing_cols:
                alter_statements.append(
                    "ALTER TABLE material ADD COLUMN contentHash VARCHAR(64) NULL"
                    if is_mysql
                    else "ALTER TABLE material ADD COLUMN contentHash VARCHAR(64)"
                )
            if "digestGeneratedAt" not in existing_cols:
                alter_statements.append(
                    "ALTER TABLE material ADD COLUMN digestGeneratedAt DATETIME NULL"
                    if is_mysql
                    else "ALTER TABLE material ADD COLUMN digestGeneratedAt DATETIME"
                )

            if alter_statements:
                with engine.begin() as conn:
                    for stmt in alter_statements:
                        conn.execute(text(stmt))

                logging.info("已自动补齐 material 表快速匹配字段: %s", ", ".join(sorted(set(alter_statements))))
    except SQLAlchemyError as exc:
        logging.exception("数据库自迁移失败（material 表快速匹配字段无法自动补齐）: %s", exc)
        raise

        raise

    # 补齐 knowledgenode 表 sortOrder 字段（手动排序）
    try:
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        if "knowledgenode" in tables:
            existing_cols = {col["name"] for col in inspector.get_columns("knowledgenode")}
            is_mysql = engine.dialect.name == "mysql"

            if "sortOrder" not in existing_cols:
                stmt = (
                    "ALTER TABLE knowledgenode ADD COLUMN sortOrder INT NOT NULL DEFAULT 0"
                    if is_mysql
                    else "ALTER TABLE knowledgenode ADD COLUMN sortOrder INTEGER NOT NULL DEFAULT 0"
                )
                with engine.begin() as conn:
                    conn.execute(text(stmt))
                logging.info("已自动补齐 knowledgenode 表 sortOrder 字段")
    except SQLAlchemyError as exc:
        logging.exception("数据库自迁移失败（knowledgenode sortOrder 字段无法自动补齐）: %s", exc)
        raise

    # 自动补齐复合索引（提升按 user/topic/parent 查询性能）
    try:
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        index_specs: list[tuple[str, str, list[str]]] = [
            ("knowledgenode", "idx_user_topic", ["userId", "topicId"]),
            ("knowledgenode", "idx_user_parent", ["userId", "parentId"]),
            ("material", "idx_material_user_topic", ["userId", "topicId"]),
        ]

        with engine.begin() as conn:
            for table, index_name, columns in index_specs:
                if table not in tables:
                    continue
                existing = {idx.get("name") for idx in inspector.get_indexes(table)}
                if index_name in existing:
                    continue
                cols_sql = ", ".join(columns)
                conn.execute(text(f"CREATE INDEX {index_name} ON {table} ({cols_sql})"))

        logging.info("已自动补齐复合索引（若缺失）")
    except SQLAlchemyError as exc:
        logging.exception("数据库自迁移失败（创建复合索引失败）: %s", exc)
        raise

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=settings.CORS_ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 健康检查
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok", "message": "Smart Tree Backend is running"}

# 数据库健康检查（用于排查“看起来生成了但没落库”的问题）
@app.get("/health/db")
async def health_check_db():
    """数据库连接与关键表检查"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        required_tables = {"question", "answerrecord"}
        missing_tables = sorted(required_tables - tables)
        return {
            "status": "ok" if not missing_tables else "degraded",
            "missingTables": missing_tables,
        }
    except SQLAlchemyError as exc:
        logging.exception("数据库健康检查失败: %s", exc)
        raise

# 包含路由
app.include_router(health.router, prefix="/api/health", tags=["Health"])
app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(topics.router, prefix="/api/topics", tags=["Topics"])
app.include_router(nodes.router, prefix="/api/nodes", tags=["Nodes"])
app.include_router(materials.router, prefix="/api/materials", tags=["Materials"])
app.include_router(questions.router, prefix="/api/questions", tags=["Questions"])
app.include_router(answer_records.router, prefix="/api/answer-records", tags=["Answer Records"])
app.include_router(node_material_links.router, prefix="/api/node-material-links", tags=["Node Material Links"])
app.include_router(node_prerequisites.router, prefix="/api/node-prerequisites", tags=["Node Prerequisites"])
app.include_router(user_notes.router, prefix="/api/user-notes", tags=["User Notes"])
app.include_router(user_settings.router, prefix="/api/user-settings", tags=["User Settings"])
app.include_router(user_profile.router, prefix="/api/user-profile", tags=["User Profile"])
app.include_router(upload.router, prefix="/api/upload", tags=["Upload"])
app.include_router(files.router, prefix="/api/files", tags=["Files"])
app.include_router(credits.router, prefix="/api/credits", tags=["Credits"])
app.include_router(llm.router, prefix="/api/llm", tags=["LLM"])
app.include_router(batches.router, prefix="/api/batches", tags=["Batch"])
app.include_router(tree.router, prefix="/api/tree", tags=["Tree Generation"])
app.include_router(community.router, prefix="/api/community", tags=["Community"])
app.include_router(shares.router, prefix="/api/shares", tags=["Shares"])
app.include_router(tree_settings.router, prefix="/api/tree-settings", tags=["Tree Settings"])
app.include_router(node_expansion.router, prefix="/api/nodes", tags=["Node Expansion"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        log_level="info"
    )
