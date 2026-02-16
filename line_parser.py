"""
line_parser.py - LINE トーク履歴 (.txt) のパーサー
"""
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

# 日付区切り行パターン
DATE_LINE_PATTERNS = [
    re.compile(r"^(\d{4}/\d{1,2}/\d{1,2})(\(.+\))?$"),
    re.compile(r"^(\d{4}年\d{1,2}月\d{1,2}日)(\(.+\))?$"),
]

# タブ区切り: 時刻\t発言者\t本文
MSG_PATTERN_TAB = re.compile(r"^(\d{1,2}:\d{2}(?::\d{2})?)\t(.+?)\t(.*)$")
# スペース区切り: 時刻  発言者  本文
MSG_PATTERN_SPACE = re.compile(r"^(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+?)\s{2,}(.*)$")
# 日付込みタブ: 日付\t時刻\t発言者\t本文
MSG_PATTERN_FULL_TAB = re.compile(
    r"^(\d{4}[/年]\d{1,2}[/月]\d{1,2}[日]?)\t(\d{1,2}:\d{2}(?::\d{2})?)\t(.+?)\t(.*)$"
)
# 日付込みスペース
MSG_PATTERN_FULL = re.compile(
    r"^(\d{4}[/年]\d{1,2}[/月]\d{1,2}[日]?)\s+(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+?)\s{2,}(.*)$"
)


@dataclass
class ParsedMessage:
    timestamp: Optional[datetime]
    speaker: str
    text: str
    raw_line: str = ""


@dataclass
class ParseResult:
    messages: list = field(default_factory=list)
    skipped_lines: int = 0
    total_lines: int = 0


def _parse_date_str(date_str: str) -> Optional[str]:
    date_str = date_str.strip()
    m = re.match(r"(\d{4})/(\d{1,2})/(\d{1,2})", date_str)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日", date_str)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return None


def _build_datetime(date_str: Optional[str], time_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        parts = time_str.split(":")
        h, mi = int(parts[0]), int(parts[1])
        s = int(parts[2]) if len(parts) > 2 else 0
        y, mo, d = [int(x) for x in date_str.split("-")]
        return datetime(y, mo, d, h, mi, s)
    except Exception:
        return None


def parse_line_txt(content: str, source_name: str = "unknown") -> ParseResult:
    result = ParseResult()
    lines = content.splitlines()
    result.total_lines = len(lines)

    current_date: Optional[str] = None
    current_msg: Optional[ParsedMessage] = None

    def _flush():
        if current_msg and current_msg.text.strip():
            result.messages.append(current_msg)

    for raw_line in lines:
        line = raw_line.rstrip()

        if not line.strip():
            _flush()
            current_msg = None
            continue

        # ヘッダー行スキップ
        if (line.startswith("[LINE]")
                or line.startswith("LINE のトーク")
                or re.match(r"^保存日時[：:]", line)
                or re.match(r"^.+のトーク履歴$", line)):
            result.skipped_lines += 1
            continue

        # 日付区切り行
        date_matched = False
        for pat in DATE_LINE_PATTERNS:
            m = pat.match(line)
            if m:
                parsed = _parse_date_str(m.group(1))
                if parsed:
                    _flush()
                    current_msg = None
                    current_date = parsed
                    date_matched = True
                    break
        if date_matched:
            continue

        # 日付+時刻込み（タブ）
        m = MSG_PATTERN_FULL_TAB.match(line)
        if m:
            _flush()
            dp = _parse_date_str(m.group(1))
            current_date = dp or current_date
            current_msg = ParsedMessage(
                timestamp=_build_datetime(current_date, m.group(2)),
                speaker=m.group(3).strip(),
                text=m.group(4).strip(),
                raw_line=raw_line,
            )
            continue

        # 日付+時刻込み（スペース）
        m = MSG_PATTERN_FULL.match(line)
        if m:
            _flush()
            dp = _parse_date_str(m.group(1))
            current_date = dp or current_date
            current_msg = ParsedMessage(
                timestamp=_build_datetime(current_date, m.group(2)),
                speaker=m.group(3).strip(),
                text=m.group(4).strip(),
                raw_line=raw_line,
            )
            continue

        # タブ区切り
        m = MSG_PATTERN_TAB.match(line)
        if m:
            _flush()
            current_msg = ParsedMessage(
                timestamp=_build_datetime(current_date, m.group(1)),
                speaker=m.group(2).strip(),
                text=m.group(3).strip(),
                raw_line=raw_line,
            )
            continue

        # スペース区切り
        m = MSG_PATTERN_SPACE.match(line)
        if m:
            _flush()
            current_msg = ParsedMessage(
                timestamp=_build_datetime(current_date, m.group(1)),
                speaker=m.group(2).strip(),
                text=m.group(3).strip(),
                raw_line=raw_line,
            )
            continue

        # 継続行
        if current_msg is not None:
            current_msg.text += "\n" + line
            continue

        result.skipped_lines += 1

    _flush()
    return result


def detect_encoding(raw_bytes: bytes) -> str:
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw_bytes.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if raw_bytes.startswith(b"\xfe\xff"):
        return "utf-16-be"
    try:
        raw_bytes.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        raw_bytes.decode("shift_jis")
        return "shift_jis"
    except UnicodeDecodeError:
        pass
    return "utf-8"


def load_line_file(raw_bytes: bytes, filename: str) -> ParseResult:
    encoding = detect_encoding(raw_bytes)
    try:
        content = raw_bytes.decode(encoding, errors="replace")
    except Exception:
        content = raw_bytes.decode("utf-8", errors="replace")
    source_name = filename.rsplit(".", 1)[0]
    return parse_line_txt(content, source_name)
