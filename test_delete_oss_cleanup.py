#!/usr/bin/env python3
"""
Regression test for "delete succeeds but OSS object still exists".

What it checks:
1) Deleting a Material that references an uploaded proxy URL (/api/files/<id>)
   also deletes the underlying UserFile record and OSS object (or local mirror).
2) Topic cascade deletion also deletes topic Materials + their files by default.

Run (in-process, no port bind; uses local-only fallback if OSS not configured):
  python test_delete_oss_cleanup.py --mode inprocess

Run (real server + real OSS):
  python test_delete_oss_cleanup.py --mode http --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
import asyncio
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx


PNG_1X1 = bytes.fromhex(
    "89504E470D0A1A0A"
    "0000000D49484452000000010000000108060000001F15C489"
    "0000000A49444154789C6360000002000154A24F5D"
    "0000000049454E44AE426082"
)


@dataclass
class Auth:
    username: str
    password: str
    token: str


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _extract_file_id(proxy_url: str) -> str:
    path = proxy_url
    if proxy_url.startswith(("http://", "https://")):
        path = urlparse(proxy_url).path or ""
    prefix = "/api/files/"
    if not path.startswith(prefix):
        raise RuntimeError(f"unexpected proxy url: {proxy_url}")
    return path[len(prefix) :]


async def _wait_until_deleted(client: httpx.AsyncClient, file_id: str, *, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    last_status: int | None = None
    while time.time() < deadline:
        resp = await client.get(f"/api/files/{file_id}")
        last_status = resp.status_code
        if resp.status_code == 404:
            return
        await asyncio.sleep(0.5)
    raise RuntimeError(f"file still readable after delete (last_status={last_status}) file_id={file_id}")


async def _register(client: httpx.AsyncClient) -> Auth:
    username = f"deltest_{uuid.uuid4().hex[:10]}"
    password = "test123456"
    resp = await client.post(
        "/api/auth/register",
        json={"username": username, "password": password, "nickname": "deltest"},
    )
    resp.raise_for_status()
    data = resp.json()
    return Auth(username=username, password=password, token=data["token"])


async def _login(client: httpx.AsyncClient, username: str, password: str) -> str:
    resp = await client.post("/api/auth/login", json={"username": username, "password": password})
    resp.raise_for_status()
    return resp.json()["token"]


async def _create_topic(client: httpx.AsyncClient, token: str) -> str:
    resp = await client.post(
        "/api/topics",
        headers=_auth_headers(token),
        json={"name": f"deltest_topic_{uuid.uuid4().hex[:6]}", "description": "deltest"},
    )
    resp.raise_for_status()
    return resp.json()["id"]


async def _create_node(client: httpx.AsyncClient, token: str, topic_id: str) -> str:
    resp = await client.post(
        "/api/nodes",
        headers=_auth_headers(token),
        json={
            "name": f"deltest_node_{uuid.uuid4().hex[:6]}",
            "description": "deltest",
            "topicId": topic_id,
            "knowledgeType": "concept",
            "difficulty": "beginner",
        },
    )
    resp.raise_for_status()
    return resp.json()["id"]


async def _upload_material_image(client: httpx.AsyncClient, token: str) -> dict[str, Any]:
    resp = await client.post(
        "/api/upload/material/image",
        headers=_auth_headers(token),
        data={"ai_analyze": "false"},
        files={"file": ("1x1.png", PNG_1X1, "image/png")},
    )
    resp.raise_for_status()
    return resp.json()


async def _upload_node_image(client: httpx.AsyncClient, token: str) -> dict[str, Any]:
    resp = await client.post(
        "/api/upload/node/image",
        headers=_auth_headers(token),
        files={"file": ("1x1.png", PNG_1X1, "image/png")},
    )
    resp.raise_for_status()
    return resp.json()


async def _create_material(
    client: httpx.AsyncClient,
    token: str,
    *,
    topic_id: str | None,
    node_id: str,
    url: str,
    file_size: int,
) -> str:
    payload: dict[str, Any] = {
        "type": "image",
        "name": f"deltest_material_{uuid.uuid4().hex[:6]}",
        "url": url,
        "fileSize": file_size,
        "nodeIds": [node_id],
    }
    if topic_id is not None:
        payload["topicId"] = topic_id
    resp = await client.post(
        "/api/materials",
        headers=_auth_headers(token),
        json=payload,
    )
    resp.raise_for_status()
    return resp.json()["id"]


async def _create_note(client: httpx.AsyncClient, token: str, *, node_id: str, content: str) -> str:
    resp = await client.post(
        "/api/user-notes",
        headers=_auth_headers(token),
        json={"nodeId": node_id, "content": content, "source": "manual"},
    )
    resp.raise_for_status()
    return resp.json()["id"]


async def _delete_node(client: httpx.AsyncClient, token: str, node_id: str) -> None:
    resp = await client.delete(f"/api/nodes/{node_id}", headers=_auth_headers(token))
    resp.raise_for_status()

async def _delete_note(client: httpx.AsyncClient, token: str, note_id: str) -> None:
    resp = await client.delete(f"/api/user-notes/{note_id}", headers=_auth_headers(token))
    resp.raise_for_status()

async def _delete_file(client: httpx.AsyncClient, token: str, file_id: str, *, force: bool = False) -> httpx.Response:
    return await client.delete(
        f"/api/files/{file_id}",
        headers=_auth_headers(token),
        params={"force": "true"} if force else None,
    )


async def _delete_material(client: httpx.AsyncClient, token: str, material_id: str) -> None:
    resp = await client.delete(f"/api/materials/{material_id}", headers=_auth_headers(token))
    resp.raise_for_status()


async def _delete_topic_cascade(client: httpx.AsyncClient, token: str, topic_id: str) -> dict[str, Any]:
    resp = await client.delete(
        f"/api/topics/{topic_id}/cascade",
        headers=_auth_headers(token),
        params={"deleteMaterials": "true", "deleteFiles": "true"},
    )
    resp.raise_for_status()
    return resp.json()

async def _clone_topic(client: httpx.AsyncClient, token: str, topic_id: str) -> dict[str, Any]:
    resp = await client.post(f"/api/topics/{topic_id}/clone", headers=_auth_headers(token), json={})
    resp.raise_for_status()
    return resp.json()

async def _get_materials_by_topic(client: httpx.AsyncClient, token: str, topic_id: str) -> list[dict[str, Any]]:
    resp = await client.get(
        "/api/materials",
        headers=_auth_headers(token),
        params={"topicId": topic_id},
    )
    resp.raise_for_status()
    return resp.json()


async def _amain() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["http", "inprocess"], default="http")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--username", default="")
    parser.add_argument("--password", default="")
    parser.add_argument(
        "--force-local",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In inprocess mode: force local uploads (no real OSS network calls)",
    )
    args = parser.parse_args()

    inprocess_app = None
    if args.mode == "inprocess":
        if args.force_local:
            os.environ.setdefault("ALLOW_LOCAL_UPLOAD_WITHOUT_OSS", "true")
            os.environ.setdefault("FORCE_LOCAL_UPLOAD", "true")
        from main import app  # noqa: WPS433 - runtime import for CLI mode

        inprocess_app = app
        await inprocess_app.router.startup()
        transport = httpx.ASGITransport(app=inprocess_app)
        client_ctx: Any = httpx.AsyncClient(transport=transport, base_url="http://inprocess")
    else:
        client_ctx = httpx.AsyncClient(base_url=args.base_url)

    try:
        async with client_ctx as client:
            # Health (best effort)
            await client.get("/health")
            await client.get("/health/db")

            if args.username:
                if not args.password:
                    raise SystemExit("--password is required when using --username")
                token = await _login(client, args.username, args.password)
                auth = Auth(username=args.username, password=args.password, token=token)
            else:
                auth = await _register(client)

            # Case 1: delete material => file disappears
            topic_id = await _create_topic(client, auth.token)
            node_id = await _create_node(client, auth.token, topic_id)
            upload = await _upload_material_image(client, auth.token)
            proxy_url = upload["url"]
            file_id = _extract_file_id(proxy_url)
            material_id = await _create_material(
                client,
                auth.token,
                topic_id=topic_id,
                node_id=node_id,
                url=proxy_url,
                file_size=int(upload.get("fileSize") or len(PNG_1X1)),
            )

            await _delete_material(client, auth.token, material_id)
            await _wait_until_deleted(client, file_id)
            print(f"[OK] delete material removed file: {file_id}")

            # Case 2: cascade delete topic => materials + files disappear
            topic_id2 = await _create_topic(client, auth.token)
            node_id2 = await _create_node(client, auth.token, topic_id2)
            upload2 = await _upload_material_image(client, auth.token)
            proxy_url2 = upload2["url"]
            file_id2 = _extract_file_id(proxy_url2)
            await _create_material(
                client,
                auth.token,
                topic_id=topic_id2,
                node_id=node_id2,
                url=proxy_url2,
                file_size=int(upload2.get("fileSize") or len(PNG_1X1)),
            )

            result = await _delete_topic_cascade(client, auth.token, topic_id2)
            await _wait_until_deleted(client, file_id2)
            print(f"[OK] topic cascade removed file: {file_id2} (result={result})")

            # Case 3: delete node => note-linked node image disappears
            topic_id3 = await _create_topic(client, auth.token)
            node_id3 = await _create_node(client, auth.token, topic_id3)
            uploaded = await _upload_node_image(client, auth.token)
            proxy_url3 = uploaded["url"]
            file_id3 = _extract_file_id(proxy_url3)
            await _create_note(client, auth.token, node_id=node_id3, content=f"![img]({proxy_url3})")
            await _delete_node(client, auth.token, node_id3)
            await _wait_until_deleted(client, file_id3)
            print(f"[OK] delete node removed note-linked file: {file_id3}")

            # Case 4: delete note => note-linked node image disappears (node still exists)
            topic_id4 = await _create_topic(client, auth.token)
            node_id4 = await _create_node(client, auth.token, topic_id4)
            uploaded4 = await _upload_node_image(client, auth.token)
            proxy_url4 = uploaded4["url"]
            file_id4 = _extract_file_id(proxy_url4)
            note_id4 = await _create_note(client, auth.token, node_id=node_id4, content=f"![img]({proxy_url4})")
            await _delete_note(client, auth.token, note_id4)
            await _wait_until_deleted(client, file_id4)
            print(f"[OK] delete note removed note-linked file: {file_id4}")

            # Case 5: delete file refuses when referenced (force works)
            topic_id5 = await _create_topic(client, auth.token)
            node_id5 = await _create_node(client, auth.token, topic_id5)
            upload5 = await _upload_material_image(client, auth.token)
            proxy_url5 = upload5["url"]
            file_id5 = _extract_file_id(proxy_url5)
            await _create_material(
                client,
                auth.token,
                topic_id=topic_id5,
                node_id=node_id5,
                url=proxy_url5,
                file_size=int(upload5.get("fileSize") or len(PNG_1X1)),
            )
            resp = await _delete_file(client, auth.token, file_id5, force=False)
            if resp.status_code != 409:
                raise RuntimeError(f"expected 409 when deleting referenced file, got {resp.status_code}: {resp.text}")
            resp_force = await _delete_file(client, auth.token, file_id5, force=True)
            if resp_force.status_code != 200:
                raise RuntimeError(f"expected 200 when force deleting file, got {resp_force.status_code}: {resp_force.text}")
            await _wait_until_deleted(client, file_id5)
            print(f"[OK] file delete blocks when referenced; force deletes: {file_id5}")

            # Case 6: clone topic => files are logically copied (no blob duplication); deleting source keeps clone readable
            # Setup source topic with a material(file)
            auth_b = await _register(client)
            src_topic_id = await _create_topic(client, auth.token)
            src_node_id = await _create_node(client, auth.token, src_topic_id)
            uploaded6 = await _upload_material_image(client, auth.token)
            src_proxy_url = uploaded6["url"]
            src_file_id = _extract_file_id(src_proxy_url)
            _src_material_id = await _create_material(
                client,
                auth.token,
                topic_id=None,
                node_id=src_node_id,
                url=src_proxy_url,
                file_size=int(uploaded6.get("fileSize") or len(PNG_1X1)),
            )

            cloned_topic = await _clone_topic(client, auth_b.token, src_topic_id)
            cloned_topic_id = cloned_topic["id"]
            cloned_materials = await _get_materials_by_topic(client, auth_b.token, cloned_topic_id)
            if not cloned_materials:
                raise RuntimeError("expected cloned materials, got empty")
            cloned_file_id = _extract_file_id(cloned_materials[0].get("url") or "")
            if cloned_file_id == src_file_id:
                raise RuntimeError("expected cloned material to reference a different file_id")

            # Delete source topic with files; cloned file should still be readable
            await _delete_topic_cascade(client, auth.token, src_topic_id)
            resp_keep = await client.get(f"/api/files/{cloned_file_id}")
            if resp_keep.status_code != 200:
                raise RuntimeError(
                    f"expected cloned file to remain after source delete, got {resp_keep.status_code}: {resp_keep.text}"
                )

            # Delete cloned material => cloned file should disappear
            await _delete_material(client, auth_b.token, cloned_materials[0]["id"])
            await _wait_until_deleted(client, cloned_file_id)
            print(f"[OK] clone topic keeps file after source delete; last ref deletes: {cloned_file_id}")
    finally:
        if inprocess_app is not None:
            await inprocess_app.router.shutdown()


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
