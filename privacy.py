"""
privacy.py - ローカル前処理（URL/メール/電話番号のマスク、ノイズ除外）
"""
import re
import unicodedata
from typing import Tuple

RE_URL = re.compile(
    r"https?://[^\s\u3000\u300a\u300b\u3001\u3002「」【】（）\[\]()]*",
    re.IGNORECASE,
)
RE_EMAIL = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
RE_PHONE_JP = re.compile(r"(?:\+81[\s\-]?)?0\d{1,4}[\s\-]?\d{1,4}[\s\-]?\d{4}")

RE_SYSTEM_MSG = re.compile(
    r"^\[?(スタンプ|写真|動画|ファイル|ボイスメッセージ|GIF|連絡先|ノート|アルバム|画像|音声|位置情報|通話|不在着信|着信拒否|コレクション)"
    r"|^[\U00010000-\U0010ffff\u2600-\u26FF\u2700-\u27BF]+$"
    r"|^\(.*\)$",
    re.UNICODE,
)

RE_EMOJI_ONLY = re.compile(
    r"^[\s\U00010000-\U0010ffff\u2600-\u26FF\u2700-\u27BF"
    r"\u00A9\u00AE\u203C\u2049\u20E3\u2122\u2139\u2194-\u2199"
    r"\u21A9\u21AA\u231A\u231B\u23E9-\u23F3\u23F8-\u23FA"
    r"\u24C2\u25AA\u25AB\u25B6\u25C0\u25FB-\u25FE\u2600-\u2604"
    r"\u260E\u2611\u2614\u2615\u2618\u261D\u2620\u2622\u2623"
    r"\u2626\u262A\u262E\u262F\u2638-\u263A\u2640\u2642\u2648-\u2653"
    r"\u265F\u2660\u2663\u2665\u2666\u2668\u267B\u267E\u267F"
    r"\u2692-\u2697\u2699\u269B\u269C\u26A0\u26A1\u26A7\u26AA"
    r"\u26AB\u26B0\u26B1\u26BD\u26BE\u26C4\u26C5\u26C8\u26CE"
    r"\u26CF\u26D1\u26D3\u26D4\u26E9\u26EA\u26F0-\u26F5\u26F7-\u26FA"
    r"\u26FD\u2702\u2705\u2708-\u270D\u270F\u2712\u2714\u2716"
    r"\u271D\u2721\u2728\u2733\u2734\u2744\u2747\u274C\u274E"
    r"\u2753-\u2755\u2757\u2763\u2764\u2795-\u2797\u27A1\u27B0"
    r"\u27BF\u2934\u2935\u2B05-\u2B07\u2B1B\u2B1C\u2B50\u2B55"
    r"\u3030\u303D\u3297\u3299]*$",
    re.UNICODE,
)


def mask_privacy(text: str) -> str:
    text = RE_URL.sub("[URL]", text)
    text = RE_EMAIL.sub("[EMAIL]", text)
    text = RE_PHONE_JP.sub("[TEL]", text)
    return text


def is_noise(text: str, min_chars: int = 2) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    char_count = len([c for c in stripped if not unicodedata.category(c).startswith("Z")])
    if char_count < min_chars:
        return True
    if RE_SYSTEM_MSG.search(stripped):
        return True
    if RE_EMOJI_ONLY.match(stripped):
        return True
    return False


def preprocess_text(text: str, min_chars: int = 2) -> Tuple[str, bool]:
    """
    戻り値: (処理済みテキスト, ノイズフラグ)
    ノイズフラグが True の場合は分析対象外
    """
    if is_noise(text, min_chars):
        return text, True
    processed = mask_privacy(text)
    return processed, False