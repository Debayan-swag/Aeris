"""
monitoring.py
Area of Interest (AOI) continuous monitoring, threshold alerts, and anomaly triggers.
"""

from typing import Any, Dict, List, Optional
from backend.database import get_db_connection


class MonitoringService:
    def create_aoi_monitor(
        self,
        name: str,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        alert_threshold: float = 0.25,
    ) -> int:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO aoi_monitors (name, min_lat, max_lat, min_lon, max_lon, alert_threshold)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (name, min_lat, max_lat, min_lon, max_lon, alert_threshold),
            )
            conn.commit()
            return cursor.lastrowid

    def list_monitors(self) -> List[Dict[str, Any]]:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM aoi_monitors ORDER BY id DESC")
            return [dict(r) for r in cursor.fetchall()]

    def check_aoi_alerts(self, aoi_id: int) -> Dict[str, Any]:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM aoi_monitors WHERE id = ?", (aoi_id,))
            monitor = cursor.fetchone()
            if not monitor:
                return {"error": "AOI monitor not found"}

            # Fetch recent images within this AOI bounding box
            cursor.execute(
                """
                SELECT * FROM images
                WHERE latitude BETWEEN ? AND ?
                  AND longitude BETWEEN ? AND ?
                ORDER BY timestamp DESC LIMIT 2
                """,
                (monitor["min_lat"], monitor["max_lat"], monitor["min_lon"], monitor["max_lon"]),
            )
            images = cursor.fetchall()
            return {
                "monitor": dict(monitor),
                "images_found": len(images),
                "status": "active",
            }


monitoring_service = MonitoringService()
