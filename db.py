import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

DB_PATH = Path("zenbu_jibun.db")


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        # テーブル新規作成
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      TEXT NOT NULL DEFAULT '',
            source       TEXT NOT NULL,
            counterparty TEXT NOT NULL,
            timestamp    TEXT,
            speaker      TEXT NOT NULL,
            is_me        INTEGER NOT NULL DEFAULT 0,
            text         TEXT NOT NULL,
            created_at   TEXT DEFAULT (datetime('now','localtime'))
        );

        CREATE TABLE IF NOT EXISTS labels (
            message_id       INTEGER PRIMARY KEY,
            style_primary    TEXT,
            think_primary    TEXT,
            style_score_json TEXT,
            think_score_json TEXT,
            FOREIGN KEY (message_id) REFERENCES messages(id)
        );

        CREATE INDEX IF NOT EXISTS idx_messages_user_id     ON messages(user_id);
        CREATE INDEX IF NOT EXISTS idx_messages_counterparty ON messages(counterparty);
        CREATE INDEX IF NOT EXISTS idx_messages_is_me        ON messages(is_me);
        CREATE INDEX IF NOT EXISTS idx_messages_source       ON messages(source);
        """)

        # 既存DBへのマイグレーション（user_id列が無ければ追加）
        cols = [
            row[1]
            for row in conn.execute("PRAGMA table_info(messages)").fetchall()
        ]
        if "user_id" not in cols:
            conn.execute(
                "ALTER TABLE messages ADD COLUMN user_id TEXT NOT NULL DEFAULT ''"
            )
            conn.commit()


def upsert_messages_batch(
    rows: List[Dict[str, Any]],
    db_path: Path = DB_PATH,
) -> List[int]:
    if not rows:
        return []

    with get_connection(db_path) as conn:
        # 同一 user_id + source の既存データを先に削除（再インポート対応）
        seen = {(r["user_id"], r["source"]) for r in rows}
        for uid, src in seen:
            # 削除対象の message_id を先に取得してラベルも消す
            old_ids = [
                r[0] for r in conn.execute(
                    "SELECT id FROM messages WHERE user_id = ? AND source = ?",
                    (uid, src),
                ).fetchall()
            ]
            if old_ids:
                conn.execute(
                    f"DELETE FROM labels WHERE message_id IN "
                    f"({','.join('?' * len(old_ids))})",
                    old_ids,
                )
            conn.execute(
                "DELETE FROM messages WHERE user_id = ? AND source = ?",
                (uid, src),
            )

        ids = []
        for row in rows:
            ts = row.get("timestamp")
            if isinstance(ts, datetime):
                ts = ts.isoformat()
            cur = conn.execute(
                """INSERT INTO messages
                   (user_id, source, counterparty, timestamp, speaker, is_me, text)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
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


def upsert_labels_batch(
    label_rows: List[Dict[str, Any]],
    db_path: Path = DB_PATH,
) -> None:
    if not label_rows:
        return
    with get_connection(db_path) as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO labels
               (message_id, style_primary, think_primary,
                style_score_json, think_score_json)
               VALUES (:message_id, :style_primary, :think_primary,
                       :style_score_json, :think_score_json)""",
            label_rows,
        )
        conn.commit()


def fetch_my_messages_with_labels(
    user_id: str,
    db_path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT m.id, m.source, m.counterparty, m.timestamp, m.speaker,
                      m.is_me, m.text,
                      l.style_primary, l.think_primary,
                      l.style_score_json, l.think_score_json
               FROM messages m
               LEFT JOIN labels l ON m.id = l.message_id
               WHERE m.is_me = 1
                 AND m.user_id = ?
               ORDER BY m.counterparty, m.timestamp""",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_sources(
    user_id: str,
    db_path: Path = DB_PATH,
) -> List[str]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT source FROM messages WHERE user_id = ? ORDER BY source",
            (user_id,),
        ).fetchall()
    return [r["source"] for r in rows]


def delete_source(
    user_id: str,
    source: str,
    db_path: Path = DB_PATH,
) -> int:
    with get_connection(db_path) as conn:
        ids = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM messages WHERE user_id = ? AND source = ?",
                (user_id, source),
            ).fetchall()
        ]
        if ids:
            conn.execute(
                f"DELETE FROM labels WHERE message_id IN "
                f"({','.join('?' * len(ids))})",
                ids,
            )
        conn.execute(
            "DELETE FROM messages WHERE user_id = ? AND source = ?",
            (user_id, source),
        )
        conn.commit()
    return len(ids)


def get_db_stats(
    user_id: str,
    db_path: Path = DB_PATH,
) -> Dict[str, int]:
    with get_connection(db_path) as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        mine = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE user_id = ? AND is_me = 1",
            (user_id,),
        ).fetchone()[0]
        labeled = conn.execute(
            """SELECT COUNT(*) FROM labels l
               JOIN messages m ON l.message_id = m.id
               WHERE m.user_id = ?""",
            (user_id,),
        ).fetchone()[0]
        sources = conn.execute(
            "SELECT COUNT(DISTINCT source) FROM messages WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0]
    return {
        "total_messages":   total,
        "my_messages":      mine,
        "labeled_messages": labeled,
        "sources":          sources,
    }