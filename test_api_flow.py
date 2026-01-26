import os
import tempfile
import uuid


def _prepare_test_env(db_path: str) -> None:
    os.environ.setdefault("JWT_SECRET", "smart-tree-test-secret")
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ.setdefault("AUTO_CREATE_TABLES", "true")
    os.environ.setdefault("AUTO_MIGRATE_SCHEMA", "true")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="smart_tree_smoke_") as tmp_dir:
        db_path = os.path.join(tmp_dir, "smart_tree_smoke.db")
        _prepare_test_env(db_path)

        from fastapi.testclient import TestClient

        import main as app_main
        from app.core.database import engine

        try:
            with TestClient(app_main.app) as client:
                resp = client.get("/health")
                assert resp.status_code == 200, resp.text

                resp = client.get("/api/community/categories")
                assert resp.status_code == 200, resp.text
                assert isinstance(resp.json(), list), resp.text

                username = f"smoke_{uuid.uuid4().hex[:8]}"
                password = "abc12345"
                resp = client.post(
                    "/api/auth/register",
                    json={
                        "username": username,
                        "password": password,
                        "confirmPassword": password,
                        "nickname": "SmokeTest",
                    },
                )
                assert resp.status_code == 200, resp.text
                token = resp.json().get("token")
                assert isinstance(token, str) and token, resp.text

                resp = client.post(
                    "/api/auth/login",
                    json={"username": username, "password": password},
                )
                assert resp.status_code == 200, resp.text
                login_token = resp.json().get("token")
                assert isinstance(login_token, str) and login_token, resp.text
                token = login_token

                resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
                assert resp.status_code == 200, resp.text

                topic_name = f"Smoke Topic {uuid.uuid4().hex[:6]}"
                resp = client.post(
                    "/api/topics",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"name": topic_name, "description": "smoke test"},
                )
                assert resp.status_code == 200, resp.text
                topic = resp.json()
                assert topic.get("name") == topic_name, resp.text

            print("smoke ok")
        finally:
            engine.dispose()


if __name__ == "__main__":
    main()
