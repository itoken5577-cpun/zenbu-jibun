#!/usr/bin/env python3
"""招待リンク生成スクリプト"""
import uuid
import sys

def generate_invite_link(base_url: str = "http://localhost:8501") -> tuple:
    uid = str(uuid.uuid4())
    link = f"{base_url}/?uid={uid}"
    return link, uid

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
    link, uid = generate_invite_link(base_url)
    
    print("=" * 60)
    print("🎫 招待リンクを生成しました")
    print("=" * 60)
    print(f"\n📎 招待リンク:\n{link}\n")
    print(f"🔑 UID:\n{uid}\n")
    print("=" * 60)
    print("📌 このリンクを共有してください。")
    print("📌 初回アクセス時にパスコード設定が求められます。")
    print("=" * 60)