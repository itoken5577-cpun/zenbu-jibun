# db.py
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

DB_PATH = Path(__file__).parent / "zenbu_jibun.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]  # r[1] = column name
    return col in cols


def init_db() -> None:
    with get_connection() as conn:
        # 1) テーブル作成（user_id は最初から持たせる）
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                source TEXT NOT NULL,
                counterparty TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                speaker TEXT NOT NULL,
                is_me INTEGER NOT NULL,
                text TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS labels (
                message_id INTEGER PRIMARY KEY,
                style_primary TEXT,
                think_primary TEXT,
                style_score_json TEXT,
                think_score_json TEXT,
                FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_user_id       ON messages(user_id);
            CREATE INDEX IF NOT EXISTS idx_messages_source        ON messages(source);
            CREATE INDEX IF NOT EXISTS idx_messages_counterparty  ON messages(counterparty);
            CREATE INDEX IF NOT EXISTS idx_messages_is_me         ON messages(is_me);
            """
        )

        # 2) 既存DB（user_id無し）からの移行も一応ケア
        #    ※ 以前に user_id 無しで作ったDBが残ってる場合向け
        if not _has_column(conn, "messages", "user_id"):
            conn.execute("ALTER TABLE messages ADD COLUMN user_id TEXT;")
            conn.execute("UPDATE messages SET user_id = 'legacy' WHERE user_id IS NULL;")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);")

        conn.commit()


def upsert_messages_batch(rows: List[Dict[str, Any]]) -> List[int]:
    if not rows:
        return []

    with get_connection() as conn:
        # 同一 user_id + source の再取り込み対応：既存分を削除
        seen = {(r["user_id"], r["source"]) for r in rows}
        for uid, src in seen:
            old_ids = [
                r[0]
                for r in conn.execute(
                    "SELECT id FROM messages WHERE user_id = ? AND source = ?",
                    (uid, src),
                ).fetchall()
            ]
            if old_ids:
                conn.execute(
                    f"DELETE FROM labels WHERE message_id IN ({','.join('?' * len(old_ids))})",
                    old_ids,
                )
            conn.execute(
                "DELETE FROM messages WHERE user_id = ? AND source = ?",
                (uid, src),
            )

        ids: List[int] = []
        for row in rows:
            ts = row.get("timestamp")
            if isinstance(ts, datetime):
                ts = ts.isoformat()

            cur = conn.execute(
                """
                INSERT INTO messages
                    (user_id, source, counterparty, timestamp, speaker, is_me, text)
                VALUES
                    (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["user_id"],
                    row["source"],
                    row["counterparty"],
                    ts,
                    row["speaker"],
                    int(row.get("is_me", 0)),
                    row["text"],
                ),
            )
            ids.append(cur.lastrowid)

        conn.commit()
        return ids


def upsert_labels_batch(label_rows: List[Dict[str, Any]]) -> None:
    if not label_rows:
        return

    with get_connection() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO labels
                (message_id, style_primary, think_primary, style_score_json, think_score_json)
            VALUES
                (:message_id, :style_primary, :think_primary, :style_score_json, :think_score_json)
            """,
            label_rows,
        )
        conn.commit()


def fetch_my_messages_with_labels(user_id: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                m.id, m.user_id, m.source, m.counterparty, m.timestamp, m.speaker, m.is_me, m.text,
                l.style_primary, l.think_primary, l.style_score_json, l.think_score_json
            FROM messages m
            LEFT JOIN labels l ON m.id = l.message_id
            WHERE m.is_me = 1 AND m.user_id = ?
            ORDER BY m.counterparty, m.timestamp
            """,
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def fetch_sources(user_id: str) -> List[str]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT source FROM messages WHERE user_id = ? ORDER BY source",
            (user_id,),
        ).fetchall()
        return [r["source"] for r in rows]


def delete_source(user_id: str, source: str) -> int:
    with get_connection() as conn:
        ids = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM messages WHERE user_id = ? AND source = ?",
                (user_id, source),
            ).fetchall()
        ]
        if ids:
            conn.execute(
                f"DELETE FROM labels WHERE message_id IN ({','.join('?' * len(ids))})",
                ids,
            )
        conn.execute(
            "DELETE FROM messages WHERE user_id = ? AND source = ?",
            (user_id, source),
        )
        conn.commit()
        return len(ids)


def get_db_stats(user_id: str) -> Dict[str, int]:
    with get_connection() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0]
        mine = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE user_id = ? AND is_me = 1",
            (user_id,),
        ).fetchone()[0]
        labeled = conn.execute(
            """
            SELECT COUNT(*)
            FROM labels l
            JOIN messages m ON l.message_id = m.id
            WHERE m.user_id = ?
            """,
            (user_id,),
        ).fetchone()[0]
        sources = conn.execute(
            "SELECT COUNT(DISTINCT source) FROM messages WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0]

        return {
            "total_messages": int(total),
            "my_messages": int(mine),
            "labeled_messages": int(labeled),
            "sources": int(sources),
        }
