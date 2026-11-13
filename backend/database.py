
import sqlite3
import json
from pathlib import Path
from contextlib import contextmanager


# ==========================================
# DATABASE CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = DATABASE_DIR / "birdvision.db"

# Directory for uploaded prediction images
HISTORY_IMAGES_DIR = PROJECT_ROOT / "uploads" / "history"
HISTORY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# DATABASE CONNECTION
# ==========================================

@contextmanager
def get_db_connection():

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


# ==========================================
# INITIALIZE DATABASE
# ==========================================

def initialize_database():

    with get_db_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                predicted_species TEXT NOT NULL,
                confidence REAL NOT NULL,
                top5_predictions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migrate existing database without deleting records.
        columns = connection.execute(
            "PRAGMA table_info(prediction_history)"
        ).fetchall()

        column_names = [column["name"] for column in columns]

        if "image_path" not in column_names:
            connection.execute("""
                ALTER TABLE prediction_history
                ADD COLUMN image_path TEXT
            """)

    print("BirdVision AI database initialized successfully.")


# ==========================================
# SAVE PREDICTION
# ==========================================

def save_prediction(
    filename,
    predicted_species,
    confidence,
    top5_predictions,
    image_path=None
):

    with get_db_connection() as connection:

        cursor = connection.execute(
            """
            INSERT INTO prediction_history (
                filename,
                predicted_species,
                confidence,
                top5_predictions,
                image_path
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                filename,
                predicted_species,
                confidence,
                json.dumps(top5_predictions),
                image_path
            )
        )

        return cursor.lastrowid


# ==========================================
# GET ALL PREDICTION HISTORY
# ==========================================

def get_prediction_history():

    with get_db_connection() as connection:

        rows = connection.execute("""
            SELECT *
            FROM prediction_history
            ORDER BY id DESC
        """).fetchall()

    history = []

    for row in rows:

        record = dict(row)

        record["top5_predictions"] = (
            json.loads(record["top5_predictions"])
            if record["top5_predictions"]
            else []
        )

        history.append(record)

    return history


# ==========================================
# DELETE ONE HISTORY RECORD
# ==========================================

def delete_prediction(prediction_id):

    with get_db_connection() as connection:

        # Get image path before deleting the record.
        row = connection.execute(
            """
            SELECT image_path
            FROM prediction_history
            WHERE id = ?
            """,
            (prediction_id,)
        ).fetchone()

        if row is None:
            return False

        image_path = row["image_path"]

        cursor = connection.execute(
            """
            DELETE FROM prediction_history
            WHERE id = ?
            """,
            (prediction_id,)
        )

    # Remove the associated image after database deletion.
    if image_path:
        image_file = (PROJECT_ROOT / image_path).resolve()

        try:
            image_file.relative_to(HISTORY_IMAGES_DIR.resolve())

            if image_file.is_file():
                image_file.unlink()

        except (ValueError, OSError):
            pass

    return cursor.rowcount > 0
