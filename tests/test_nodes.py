"""
Tests for knowledge nodes endpoints.
"""
import pytest
from fastapi.testclient import TestClient


class TestNodes:
    """Test knowledge nodes functionality."""

    @pytest.fixture
    def setup_topic(self, client: TestClient, test_user_data, test_topic_data):
        """Setup: Create user and topic."""
        register_response = client.post("/api/auth/register", json=test_user_data)
        token = register_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        topic_response = client.post("/api/topics", json=test_topic_data, headers=headers)
        topic_id = topic_response.json()["id"]
        return {"headers": headers, "topic_id": topic_id, "token": token}

    def test_create_node(self, client: TestClient, test_node_data, setup_topic):
        """Test creating a knowledge node."""
        headers = setup_topic["headers"]
        topic_id = setup_topic["topic_id"]

        node_data = {**test_node_data, "topicId": topic_id}
        response = client.post("/api/nodes", json=node_data, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == test_node_data["name"]
        assert "id" in data

    def test_get_nodes_by_topic(self, client: TestClient, test_node_data, setup_topic):
        """Test retrieving nodes for a topic."""
        headers = setup_topic["headers"]
        topic_id = setup_topic["topic_id"]

        # Create node
        node_data = {**test_node_data, "topicId": topic_id}
        client.post("/api/nodes", json=node_data, headers=headers)

        # Get nodes
        response = client.get(f"/api/nodes?topicId={topic_id}", headers=headers)
        assert response.status_code == 200
        nodes = response.json()
        assert isinstance(nodes, list)
        assert len(nodes) >= 1

    def test_update_node(self, client: TestClient, test_node_data, setup_topic):
        """Test updating a node."""
        headers = setup_topic["headers"]
        topic_id = setup_topic["topic_id"]

        # Create node
        node_data = {**test_node_data, "topicId": topic_id}
        create_response = client.post("/api/nodes", json=node_data, headers=headers)
        node_id = create_response.json()["id"]

        # Update node
        updated_data = {"description": "更新后的描述"}
        response = client.put(f"/api/nodes/{node_id}", json=updated_data, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["description"] == "更新后的描述"

    def test_delete_node(self, client: TestClient, test_node_data, setup_topic):
        """Test deleting a node."""
        headers = setup_topic["headers"]
        topic_id = setup_topic["topic_id"]

        # Create node
        node_data = {**test_node_data, "topicId": topic_id}
        create_response = client.post("/api/nodes", json=node_data, headers=headers)
        node_id = create_response.json()["id"]

        # Delete node
        response = client.delete(f"/api/nodes/{node_id}", headers=headers)
        assert response.status_code == 200

    def test_create_child_node(self, client: TestClient, test_node_data, setup_topic):
        """Test creating a child node with parent relationship."""
        headers = setup_topic["headers"]
        topic_id = setup_topic["topic_id"]

        # Create parent node
        parent_data = {**test_node_data, "topicId": topic_id}
        parent_response = client.post("/api/nodes", json=parent_data, headers=headers)
        parent_id = parent_response.json()["id"]

        # Create child node
        child_data = {
            **test_node_data,
            "name": "梯度下降",
            "topicId": topic_id,
            "parentId": parent_id
        }
        response = client.post("/api/nodes", json=child_data, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["parentId"] == parent_id
