#!/usr/bin/env python3
"""
Audit (and optionally delete) orphaned uploaded files.

Orphan definition:
  user_file row exists but is not referenced by:
    - material.url == "/api/files/<id>"
    - user_note.content contains "/api/files/<id>"

Notes:
  - This script does NOT parse raw OSS URLs stored elsewhere; it's for UserFile-backed uploads.
  - Use --delete to also delete OSS/local mirror + remove DB rows.

Examples:
  python audit_orphan_files.py --mode inprocess
  python audit_orphan_files.py --mode inprocess --delete
  python audit_orphan_files.py --mode inprocess --username alice --delete
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass
from typing import Any

import httpx


async def _amain() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inprocess"], default="inprocess")
    parser.add_argument("--username", default="")
    parser.add_argument("--delete", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    inprocess_app = None
    # Allow running without external OSS, using local uploads mirror if present.
    os.environ.setdefault("FORCE_LOCAL_UPLOAD", "true")
    from main import app  # noqa: WPS433

    inprocess_app = app
    await inprocess_app.router.startup()
    transport = httpx.ASGITransport(app=inprocess_app)
    client_ctx: Any = httpx.AsyncClient(transport=transport, base_url="http://inprocess")

    try:
        async with client_ctx as client:  # noqa: F841 - ensures ASGI lifespan is active
            if not args.username:
                raise SystemExit("--username is required (we need user scope to audit/delete)")

            from app.core.database import SessionLocal  # noqa: WPS433
            from app.core.schema import ensure_user_file_schema  # noqa: WPS433
            from app.core.oss import delete_file_by_key  # noqa: WPS433
            from app.models import Material, UserFile, UserNote, User  # noqa: WPS433

            db = SessionLocal()
            try:
                ensure_user_file_schema(db)
                me = db.query(User).filter(User.username == args.username).first()
                if not me:
                    raise SystemExit("user not found in DB")

                files = db.query(UserFile).filter(UserFile.userId == me.id).all()
                orphan_ids: list[str] = []
                for f in files:
                    proxy_url = f"/api/files/{f.id}"
                    material_refs = db.query(Material).filter(Material.userId == me.id, Material.url == proxy_url).count()
                    note_refs = (
                        db.query(UserNote)
                        .filter(UserNote.userId == me.id, UserNote.content.contains(proxy_url))
                        .count()
                    )
                    if material_refs or note_refs:
                        continue
                    orphan_ids.append(f.id)

                print(f"Total files: {len(files)}  Orphans: {len(orphan_ids)}")
                for fid in orphan_ids[:50]:
                    print(f"- orphan: {fid}")
                if len(orphan_ids) > 50:
                    print(f"... and {len(orphan_ids) - 50} more")

                if args.delete and orphan_ids:
                    deleted = 0
                    for fid in orphan_ids:
                        rec = db.query(UserFile).filter(UserFile.id == fid, UserFile.userId == me.id).first()
                        if not rec:
                            continue
                        ok = delete_file_by_key(rec.ossPath)
                        if not ok:
                            print(f"[WARN] failed to delete object for {fid}: {rec.ossPath}")
                            continue
                        db.delete(rec)
                        deleted += 1
                    db.commit()
                    print(f"Deleted orphans: {deleted}")
            finally:
                db.close()
    finally:
        if inprocess_app is not None:
            await inprocess_app.router.shutdown()

    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
