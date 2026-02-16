"""
app.py - 全部自分 MVP
コミュニケーションスタイルと思考スタイルの可視化アプリ
"""
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
import altair as alt

from line_parser import load_line_file
from privacy import preprocess_text
from classify_rules import (
    classify_to_json,
    COMM_STYLE_LABELS,
    THINK_STYLE_LABELS,
    COMM_STYLE_DISPLAY,
    THINK_STYLE_DISPLAY,
)


from db import (
    init_db, upsert_messages_batch, upsert_labels_batch,
    fetch_my_messages_with_labels, fetch_sources, get_db_stats, delete_source,
)
from aggregate import (
    build_distribution, calc_diff_from_global, top3_diff,
    build_summary_json, dist_to_dataframe,
)

# ─────────────────────────────────────
# ページ設定
# ─────────────────────────────────────
st.set_page_config(
    page_title="全部自分",
    page_icon="🪞",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# ─────────────────────────────────────
# CSS
# ─────────────────────────────────────
st.markdown("""
<style>
.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.sub-title { font-size:1rem; color:#888; margin-bottom:1.2rem; }
.metric-card {
    background:#f8f9ff; border:1px solid #e8eaf6;
    border-radius:12px; padding:1rem 1.2rem;
    margin:0.3rem 0; text-align:center;
}
.privacy-badge {
    background:#e8f5e9; color:#2e7d32; border-radius:20px;
    padding:0.2rem 0.9rem; font-size:0.82rem;
    display:inline-block; margin-bottom:1.2rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────
# ヘルパー関数
# ─────────────────────────────────────

def process_files(file_pairs: List[tuple], my_name: str, min_chars: int) -> None:
    total_labeled = 0
    total_skip_lines = 0
    results_info = []

    progress = st.progress(0.0)
    status_ph = st.empty()

    for i, (raw_bytes, filename) in enumerate(file_pairs):
        progress.progress((i + 1) / len(file_pairs))
        status_ph.text(f"処理中: {filename}  ({i+1}/{len(file_pairs)})")

        source = filename
        counterparty = filename.rsplit(".", 1)[0]

        parse_result = load_line_file(raw_bytes, filename)

        msg_rows: List[Dict[str, Any]] = []
        meta: List[tuple] = []

        for pm in parse_result.messages:
            is_me = 1 if pm.speaker == my_name else 0
            proc_text, noise_flag = preprocess_text(pm.text, min_chars)
            msg_rows.append({
                "source": source,
                "counterparty": counterparty,
                "timestamp": pm.timestamp,
                "speaker": pm.speaker,
                "is_me": is_me,
                "text": pm.text,
            })
            meta.append((is_me, proc_text, noise_flag))

        ids = upsert_messages_batch(msg_rows)

        label_batch = []
        noise_count = 0
        for row_id, (is_me, proc_text, noise_flag) in zip(ids, meta):
            if noise_flag:
                noise_count += 1
            if is_me and not noise_flag:
                clf = classify_to_json(proc_text)
                label_batch.append({"message_id": row_id, **clf})

        upsert_labels_batch(label_batch)

        n_mine = sum(1 for im, _, _ in meta if im)
        results_info.append({
            "ファイル名": filename,
            "トークルーム": counterparty,
            "総メッセージ": len(parse_result.messages),
            "自分の発言": n_mine,
            "分析対象": len(label_batch),
            "ノイズ除外": noise_count,
            "スキップ行": parse_result.skipped_lines,
        })
        total_labeled += len(label_batch)
        total_skip_lines += parse_result.skipped_lines

    progress.progress(1.0)
    status_ph.empty()

    st.success(f"✅ 取り込み完了！ 分析対象メッセージ: {total_labeled} 件")
    if results_info:
        st.dataframe(pd.DataFrame(results_info), use_container_width=True, hide_index=True)
    if total_skip_lines > 0:
        st.caption(f"ℹ️ パースできなかった行: 合計 {total_skip_lines} 行（ヘッダー・システムメッセージ等）")


def render_grouped_bar(df: pd.DataFrame, labels: List[str]) -> None:
    df_reset = df[labels].reset_index()
    df_melt = df_reset.melt(id_vars="counterparty", var_name="ラベル", value_name="比率")
    order = ["global"] + [c for c in df_reset["counterparty"].tolist() if c != "global"]

    chart = (
        alt.Chart(df_melt)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("ラベル:N", sort=labels, title=None),
            y=alt.Y("比率:Q", axis=alt.Axis(format=".0%"), title="割合"),
            color=alt.Color("counterparty:N", legend=alt.Legend(title="トークルーム"), sort=order),
            xOffset="counterparty:N",
            tooltip=[
                alt.Tooltip("counterparty:N", title="トークルーム"),
                alt.Tooltip("ラベル:N", title="ラベル"),
                alt.Tooltip("比率:Q", title="割合", format=".1%"),
            ],
        )
        .properties(height=270)
        .configure_axis(labelFontSize=12)
    )
    st.altair_chart(chart, use_container_width=True)


def render_compare_bar(
    cp_dist: Dict, global_dist: Dict, labels: List[str], cp_name: str
) -> None:
    rows = []
    for label in labels:
        rows.append({"ラベル": label, "値": float(cp_dist.get(label, 0)), "種別": f"「{cp_name}」"})
        rows.append({"ラベル": label, "値": float(global_dist.get(label, 0)), "種別": "全体平均"})
    df = pd.DataFrame(rows)

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("ラベル:N", sort=labels, title=None),
            y=alt.Y("値:Q", axis=alt.Axis(format=".0%"), title="割合"),
            color=alt.Color(
                "種別:N",
                scale=alt.Scale(
                    domain=[f"「{cp_name}」", "全体平均"],
                    range=["#667eea", "#cccccc"],
                ),
                legend=alt.Legend(title=""),
            ),
            xOffset="種別:N",
            tooltip=[
                alt.Tooltip("ラベル:N", title="ラベル"),
                alt.Tooltip("種別:N"),
                alt.Tooltip("値:Q", title="割合", format=".1%"),
            ],
        )
        .properties(height=240)
    )
    st.altair_chart(chart, use_container_width=True)


# ─────────────────────────────────────
# ページヘッダー
# ─────────────────────────────────────
st.markdown('<div class="main-title">🪞 全部自分</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">相手ごとに、自分のコミュニケーション・思考スタイルの違いを可視化する</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="privacy-badge">🔒 生ログは外部送信しません（ローカル完結）</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────
# サイドバー
# ─────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 設定")

    my_name = st.text_input(
        "自分の表示名",
        value=st.session_state.get("my_name", ""),
        placeholder="例：健悟",
        help="LINEトーク履歴に表示される自分の名前",
    )
    if my_name:
        st.session_state["my_name"] = my_name

    min_chars = st.slider("ノイズ除外・最小文字数", 1, 10, 2)

    st.divider()
    st.subheader("📊 DB 統計")
    stats = get_db_stats()
    st.metric("総メッセージ", stats["total_messages"])
    st.metric("自分のメッセージ", stats["my_messages"])
    st.metric("ラベル付き", stats["labeled_messages"])
    st.metric("ファイル数", stats["sources"])

    sources = fetch_sources()
    if sources:
        st.divider()
        st.subheader("🗑️ データ削除")
        del_src = st.selectbox(
            "削除するソース", ["（選択してください）"] + sources, key="del_src_sel"
        )
        if del_src != "（選択してください）":
            if st.button(f"「{del_src}」を削除", type="secondary"):
                cnt = delete_source(del_src)
                st.success(f"{cnt} 件削除しました")
                st.rerun()

# ─────────────────────────────────────
# タブ
# ─────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📥 取り込み", "📊 分析・可視化", "📤 LLM エクスポート"])

# =============================================
# Tab 1: 取り込み
# =============================================
with tab1:
    st.header("LINEトーク履歴の取り込み")

    if not my_name:
        st.warning("⚠️ まずサイドバーで「自分の表示名」を入力してください")
    else:
        st.info(
            "**使い方：** LINEアプリ → トーク → メニュー → トーク履歴を送信 → `.txt` を保存  \n"
            "1ファイル = 1トークルームとして扱います。ファイル名がトークルーム名になります。"
        )

        uploaded_files = st.file_uploader(
            "LINEトーク履歴 (.txt) をドラッグ＆ドロップ",
            type=["txt"],
            accept_multiple_files=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if uploaded_files:
                if st.button("🚀 取り込む", type="primary", use_container_width=True):
                    pairs = [(f.read(), f.name) for f in uploaded_files]
                    process_files(pairs, my_name, min_chars)
                    st.rerun()
        with col_b:
            if st.button("🧪 サンプルデータで試す", use_container_width=True):
                sample_path = Path(__file__).parent / "sample_data" / "sample_line.txt"
                if sample_path.exists():
                    raw = sample_path.read_bytes()
                    _name = my_name or "健悟"
                    if not my_name:
                        st.session_state["my_name"] = "健悟"
                    process_files([(raw, "sample_line.txt")], _name, min_chars)
                    st.rerun()
                else:
                    st.error("sample_data/sample_line.txt が見つかりません")

# =============================================
# Tab 2: 分析・可視化
# =============================================
with tab2:
    st.header("コミュニケーション & 思考スタイル分析")

    messages = fetch_my_messages_with_labels()

    if not messages:
        st.info("データがありません。「取り込み」タブで LINEログを取り込んでください。")
    else:
        dist_result = build_distribution(messages)
        diffs_all = calc_diff_from_global(dist_result)
        df_style, df_think = dist_to_dataframe(dist_result)

        g = dist_result.get("global", {})
        counterparties = [cp for cp in dist_result.keys() if cp != "global"]

        # ── サマリーカード ──
        st.subheader("📈 全体サマリー")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("分析メッセージ数", g.get("count", 0))
        with c2:
            st.metric("トークルーム数", len(counterparties))
        with c3:
            sd = g.get("style_dist", {})
            ts = max(sd, key=sd.get) if sd else "—"
            st.metric("最頻コミュスタイル", ts, f"{sd.get(ts, 0):.0%}")
        with c4:
            td = g.get("think_dist", {})
            tt = max(td, key=td.get) if td else "—"
            st.metric("最頻思考スタイル", tt, f"{td.get(tt, 0):.0%}")

        # ── 全相手の比較グラフ ──
        st.subheader("🎨 コミュニケーションスタイル分布")
        render_grouped_bar(df_style.rename(columns=COMM_STYLE_DISPLAY), [COMM_STYLE_DISPLAY[k] for k in COMM_STYLE_LABELS])

        st.subheader("🧠 思考スタイル分布")
        render_grouped_bar(df_think.rename(columns=THINK_STYLE_DISPLAY), [THINK_STYLE_DISPLAY[k] for k in THINK_STYLE_LABELS])

        # ── 相手別詳細 ──
        st.divider()
        st.subheader("👤 相手別スタイル詳細")

        if not counterparties:
            st.info("相手別データがありません")
        else:
            sel = st.selectbox(
                "分析するトークルームを選択",
                ["（全相手を比較）"] + counterparties,
            )

            if sel == "（全相手を比較）":
                tab_cs, tab_ts = st.tabs(["コミュニケーションスタイル", "思考スタイル"])
                with tab_cs:
                    disp = df_style[COMM_STYLE_LABELS].map(lambda x: f"{float(x):.1%}")
                    st.dataframe(disp, use_container_width=True)
                with tab_ts:
                    disp = df_think[THINK_STYLE_LABELS].map(lambda x: f"{float(x):.1%}")
                    st.dataframe(disp, use_container_width=True)
            else:
                cp_data = dist_result.get(sel, {})
                g_data = dist_result.get("global", {})

                st.markdown(f"### 📌 「{sel}」 との会話")
                st.caption(f"メッセージ数: {cp_data.get('count', 0)} 件")

                t3 = top3_diff(diffs_all, sel)
                if t3:
                    st.markdown("#### 🔍 全体平均との差分 Top3")
                    cols3 = st.columns(3)
                    for i, item in enumerate(t3):
                        dv = item["diff"]
                        sign, color = ("▲", "#e74c3c") if dv > 0 else ("▼", "#3498db")
                        with cols3[i]:
                            st.markdown(
                                f"""<div class="metric-card">
                                <div style="font-size:.75rem;color:#888;">{item['kind']}スタイル</div>
                                <div style="font-size:1.3rem;font-weight:bold;">{item['label']}</div>
                                <div style="font-size:1.05rem;color:{color};">{sign}{abs(dv):.1%}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                st.markdown("#### コミュニケーションスタイル 比較")
                render_compare_bar(
                    cp_data.get("style_dist", {}),
                    g_data.get("style_dist", {}),
                    COMM_STYLE_LABELS,
                    sel,
                )
                st.markdown("#### 思考スタイル 比較")
                render_compare_bar(
                    cp_data.get("think_dist", {}),
                    g_data.get("think_dist", {}),
                    THINK_STYLE_LABELS,
                    sel,
                )

# =============================================
# Tab 3: LLM エクスポート
# =============================================
with tab3:
    st.header("外部 LLM へのエクスポート（任意）")

    st.markdown("""
**🔒 プライバシー：** 生メッセージは含まれません（集計比率データのみ）  
気になる場合は JSON の `display_name` を手動削除してから貼り付けてください。

**使い方：** JSON を生成 → コピー → ChatGPT/Claude に「プロンプト」と共に貼り付け
""")


    _my_name = st.session_state.get("my_name", "ユーザー")
    msgs_exp = fetch_my_messages_with_labels()

    if not msgs_exp:
        st.info("データがありません。「取り込み」タブで LINEログを取り込んでください。")
    else:
        if st.button("📋 集計 JSON を生成", type="primary"):
            dr = build_distribution(msgs_exp)
            dfs2 = calc_diff_from_global(dr)
            summary = build_summary_json(dr, dfs2, _my_name)
            st.session_state["summary_json"] = json.dumps(summary, ensure_ascii=False, indent=2)

        if "summary_json" in st.session_state:
            st.success("✅ 生成完了（生ログは含まれていません）")

            st.text_area(
                "集計 JSON（コピーして LLM に貼り付けてください）",
                value=st.session_state["summary_json"],
                height=380,
            )
            st.download_button(
                label="⬇️ JSON をダウンロード",
                data=st.session_state["summary_json"].encode("utf-8"),
                file_name="zenbu_jibun_summary.json",
                mime="application/json",
            )

            st.divider()
            st.subheader("📝 インサイト生成プロンプト")
            prompt_path = Path(__file__).parent / "prompts" / "insight_prompt.txt"
            if prompt_path.exists():
                st.text_area(
                    "このプロンプトの後に上記 JSON を貼り付けて LLM に送信してください",
                    value=prompt_path.read_text(encoding="utf-8"),
                    height=300,
                )
            else:
                st.warning("prompts/insight_prompt.txt が見つかりません")