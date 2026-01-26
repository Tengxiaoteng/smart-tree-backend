"""
Pytest configuration and fixtures for backend tests.
"""
import os
import sys
from typing import Generator
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import Base, get_db
from app.core.auth import create_access_token
from main import app

# Use in-memory SQLite for tests
TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    """Create a fresh database session for each test."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session: Session) -> TestClient:
    """Create a test client with overridden database dependency."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def test_user_data():
    """Sample user data for testing."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "TestPass123!"
    }


@pytest.fixture
def auth_headers(test_user_data):
    """Generate authentication headers for testing."""
    token = create_access_token(data={"sub": test_user_data["username"]})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def test_topic_data():
    """Sample topic data for testing."""
    return {
        "name": "机器学习",
        "description": "机器学习基础知识树",
        "scope": ["机器学习", "深度学习", "神经网络"],
        "keywords": ["AI", "ML", "算法"]
    }


@pytest.fixture
def test_node_data():
    """Sample knowledge node data for testing."""
    return {
        "name": "线性回归",
        "description": "线性回归是一种基本的监督学习算法",
        "learningObjectives": ["理解线性回归原理", "掌握梯度下降"],
        "keyConcepts": ["损失函数", "梯度下降", "正则化"],
        "knowledgeType": "concept",
        "difficulty": "beginner",
        "estimatedMinutes": 30,
        "prerequisites": [],
        "questionPatterns": [],
        "commonMistakes": []
    }


@pytest.fixture
def test_material_data():
    """Sample material data for testing."""
    return {
        "title": "线性回归教程",
        "content": "线性回归是机器学习中最基础的算法之一...",
        "type": "text",
        "source": "manual",
        "url": "https://example.com/tutorial"
    }
