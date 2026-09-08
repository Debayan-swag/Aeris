"""
timeline.py
Temporal query and satellite observation timeline generation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from backend.database import get_db_connection


def get_timeline_by_location(
    latitude: float,
    longitude: float,
    radius_km: float = 10.0,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve satellite image timeline for a specific geographic coordinate bounding area.
    """
    # Rough approximation: 1 deg lat ~ 111km, 1 deg lon ~ 111km * cos(lat)
    deg_delta = radius_km / 111.0

    min_lat = latitude - deg_delta
    max_lat = latitude + deg_delta
    min_lon = longitude - deg_delta
    max_lon = longitude + deg_delta

    query = """
        SELECT * FROM images
        WHERE latitude BETWEEN ? AND ?
          AND longitude BETWEEN ? AND ?
    """
    params: list = [min_lat, max_lat, min_lon, max_lon]

    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date.isoformat())
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date.isoformat())

    query += " ORDER BY timestamp ASC"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
