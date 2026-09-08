"""
create_database.py
Initializes the SQLite database schema for TerraWatch metadata storage.
"""

import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "database"
DB_PATH = DB_DIR / "terrawatch.db"


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Initializing database at: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Images table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_no INTEGER UNIQUE,
        image_name TEXT,
        file_path TEXT NOT NULL,
        width INTEGER,
        height INTEGER,
        format TEXT,
        latitude REAL,
        longitude REAL,
        timestamp DATETIME,
        cloud_cover REAL,
        source TEXT DEFAULT 'sentinel-2',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Vector index mapping
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS image_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER NOT NULL,
        embedding_index INTEGER NOT NULL,
        model_name TEXT NOT NULL,
        vector_dim INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
    );
    """)

    # Change detection events
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS change_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_before_id INTEGER NOT NULL,
        image_after_id INTEGER NOT NULL,
        change_score REAL,
        change_type TEXT,
        change_mask_path TEXT,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (image_before_id) REFERENCES images(id),
        FOREIGN KEY (image_after_id) REFERENCES images(id)
    );
    """)

    # Monitoring AOI (Areas of Interest)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS aoi_monitors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        min_lat REAL,
        max_lat REAL,
        min_lon REAL,
        max_lon REAL,
        alert_threshold REAL DEFAULT 0.25,
        status TEXT DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    conn.close()
    print("Database schema successfully created.")


if __name__ == "__main__":
    init_db()
