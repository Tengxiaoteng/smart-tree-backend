"""
Tests for authentication endpoints.
"""
import pytest
from fastapi.testclient import TestClient


class TestAuth:
    """Test authentication functionality."""

    def test_register_user(self, client: TestClient, test_user_data):
        """Test user registration."""
        response = client.post("/api/auth/register", json=test_user_data)
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_register_duplicate_user(self, client: TestClient, test_user_data):
        """Test registering duplicate username."""
        # First registration
        client.post("/api/auth/register", json=test_user_data)
        # Second registration with same username
        response = client.post("/api/auth/register", json=test_user_data)
        assert response.status_code == 400

    def test_login_success(self, client: TestClient, test_user_data):
        """Test successful login."""
        # Register first
        client.post("/api/auth/register", json=test_user_data)
        # Then login
        response = client.post("/api/auth/login", json={
            "username": test_user_data["username"],
            "password": test_user_data["password"]
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, client: TestClient, test_user_data):
        """Test login with wrong password."""
        # Register first
        client.post("/api/auth/register", json=test_user_data)
        # Login with wrong password
        response = client.post("/api/auth/login", json={
            "username": test_user_data["username"],
            "password": "WrongPassword123!"
        })
        assert response.status_code == 401

    def test_login_nonexistent_user(self, client: TestClient):
        """Test login with non-existent user."""
        response = client.post("/api/auth/login", json={
            "username": "nonexistent",
            "password": "Password123!"
        })
        assert response.status_code == 401
