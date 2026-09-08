"""
database.py
Database connection manager and query helpers for TerraWatch SQLite store.
"""

import sqlite3
from typing import Any, Dict, List, Optional
from backend.config import settings


def get_db_connection() -> sqlite3.Connection:
    settings.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_image_by_id(image_id: int) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def list_images(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM images ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))
        return [dict(row) for row in cursor.fetchall()]


def record_change_event(
    image_before_id: int,
    image_after_id: int,
    change_score: float,
    change_type: str,
    change_mask_path: Optional[str] = None,
    description: Optional[str] = None,
) -> int:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO change_events (
                image_before_id, image_after_id, change_score, change_type, change_mask_path, description
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (image_before_id, image_after_id, change_score, change_type, change_mask_path, description),
        )
        conn.commit()
        return cursor.lastrowid
