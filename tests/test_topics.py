"""
Tests for topics (knowledge trees) endpoints.
"""
import pytest
from fastapi.testclient import TestClient


class TestTopics:
    """Test topics/knowledge trees functionality."""

    def test_create_topic_authenticated(self, client: TestClient, test_user_data, test_topic_data):
        """Test creating a topic with authentication."""
        # Register and login
        register_response = client.post("/api/auth/register", json=test_user_data)
        token = register_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Create topic
        response = client.post("/api/topics", json=test_topic_data, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == test_topic_data["name"]
        assert data["description"] == test_topic_data["description"]
        assert "id" in data

    def test_create_topic_unauthenticated(self, client: TestClient, test_topic_data):
        """Test creating a topic without authentication."""
        response = client.post("/api/topics", json=test_topic_data)
        assert response.status_code == 401

    def test_get_topics(self, client: TestClient, test_user_data, test_topic_data):
        """Test retrieving user's topics."""
        # Register and create topic
        register_response = client.post("/api/auth/register", json=test_user_data)
        token = register_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        client.post("/api/topics", json=test_topic_data, headers=headers)

        # Get topics
        response = client.get("/api/topics", headers=headers)
        assert response.status_code == 200
        topics = response.json()
        assert isinstance(topics, list)
        assert len(topics) >= 1
        assert topics[0]["name"] == test_topic_data["name"]

    def test_update_topic(self, client: TestClient, test_user_data, test_topic_data):
        """Test updating a topic."""
        # Register and create topic
        register_response = client.post("/api/auth/register", json=test_user_data)
        token = register_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        create_response = client.post("/api/topics", json=test_topic_data, headers=headers)
        topic_id = create_response.json()["id"]

        # Update topic
        updated_data = {"name": "深度学习", "description": "深度学习知识树"}
        response = client.put(f"/api/topics/{topic_id}", json=updated_data, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "深度学习"
        assert data["description"] == "深度学习知识树"

    def test_delete_topic(self, client: TestClient, test_user_data, test_topic_data):
        """Test deleting a topic."""
        # Register and create topic
        register_response = client.post("/api/auth/register", json=test_user_data)
        token = register_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        create_response = client.post("/api/topics", json=test_topic_data, headers=headers)
        topic_id = create_response.json()["id"]

        # Delete topic
        response = client.delete(f"/api/topics/{topic_id}", headers=headers)
        assert response.status_code == 200

        # Verify deletion
        get_response = client.get("/api/topics", headers=headers)
        topics = get_response.json()
        assert len(topics) == 0
