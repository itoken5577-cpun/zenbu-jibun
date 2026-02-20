"""
app.py - Prismee MVP（招待リンク＋ロック版）
"""
import json
import uuid
import re
import os
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
import altair as alt
import plotly.graph_objects as go
import streamlit.components.v1 as components

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
    get_user_auth_state, set_passcode, verify_passcode,
)
from aggregate import (
    build_distribution, calc_diff_from_global, top3_diff,
    build_summary_json, dist_to_dataframe,
)

# ─────────────────────────────────────
# ページ設定
# ─────────────────────────────────────
st.set_page_config(
    page_title="Prismee",
    page_icon="🪞",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# ─────────────────────────────────────
# 招待リンク必須化チェック
# ─────────────────────────────────────
def validate_uid(uid: str | None) -> str | None:
    """uid のバリデーション"""
    if not uid or not uid.strip():
        return None
    uid = uid.strip()
    if not re.match(r"^[a-f0-9\-]{8,}$", uid, re.IGNORECASE):
        return None
    return uid


uid_param = st.query_params.get("uid")
USER_ID = validate_uid(uid_param)

if not USER_ID:
    st.markdown(
        """
        <div style="
            max-width:600px; margin:100px auto; padding:40px;
            background:#fff3cd; border-radius:16px;
            text-align:center; border:2px solid #ffc107;
        ">
            <h2 style="color:#856404;">🔒 招待リンクが必要です</h2>
            <p style="font-size:1.1rem; color:#856404; line-height:1.8;">
                このアプリは招待リンク経由でのみアクセスできます。<br>
                招待リンク（<code>?uid=...</code>）を受け取ってアクセスしてください。
            </p>
            <hr style="border:none; border-top:1px solid #ffc107; margin:20px 0;">
            <p style="font-size:0.9rem; color:#856404;">
                💡 招待リンクの発行は管理者にお問い合わせください。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ─────────────────────────────────────
# 認証ゲート（パスコード）
# ─────────────────────────────────────
auth_state = get_user_auth_state(USER_ID)
is_authenticated = st.session_state.get("authed_uid") == USER_ID

if not is_authenticated:
    st.markdown(
        """
        <div style="max-width:500px; margin:80px auto;">
            <h2 style="text-align:center;">🔐 パスコード認証</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if auth_state["locked_until"]:
        st.error(f"⏱️ ロック中です（解除時刻: {auth_state['locked_until'][:16]}）")
        st.info("しばらく待ってから再度アクセスしてください。")
        st.stop()

    if not auth_state["has_pass"]:
        st.info("✨ 初回アクセスです。このスペース専用のパスコードを設定してください。")
        
        with st.form("set_passcode_form"):
            new_pass = st.text_input("パスコード（4桁以上の数字を推奨）", type="password", max_chars=20)
            new_pass_confirm = st.text_input("パスコード（確認）", type="password", max_chars=20)
            submit = st.form_submit_button("設定する", type="primary")

        if submit:
            if len(new_pass) < 4:
                st.error("パスコードは4文字以上にしてください")
            elif new_pass != new_pass_confirm:
                st.error("パスコードが一致しません")
            else:
                set_passcode(USER_ID, new_pass)
                st.success("✅ パスコードを設定しました")
                st.session_state["authed_uid"] = USER_ID
                st.rerun()
    else:
        st.info("🔓 パスコードを入力してロック解除してください")
        
        with st.form("unlock_form"):
            input_pass = st.text_input("パスコード", type="password", max_chars=20)
            submit = st.form_submit_button("解除", type="primary")

        if submit:
            result = verify_passcode(USER_ID, input_pass)
            if result["success"]:
                st.success("✅ 認証成功")
                st.session_state["authed_uid"] = USER_ID
                st.rerun()
            else:
                st.error(f"❌ {result['message']}")

    st.stop()

# ─────────────────────────────────────
# ここから先は認証済みユーザーのみ
# ─────────────────────────────────────

# スタイルガイド データ定義
COMM_STYLE_GUIDE = {
    "Lead_Directiveness": {
        "ja_name": "主導性",
        "emoji": "🧭",
        "desc": "方向性を示し、決める・依頼する・前に進める発言が多い軸です。",
        "scenes": ["意思決定が必要なとき", "タスクを明確にしたいとき", "停滞を前進させたいとき"],
        "value": "会話を行動に接続し、進行を生みます。",
        "questions": ["主導するのは“前に進めたい”から？それとも“不安”から？", "相手が動きやすい依頼の出し方になっていますか？"],
        "high": "推進力が強く、決まる・進む会話になりやすい。",
        "low": "合意形成や意思決定が遅れやすい（流れに委ねがち）。",
        "tips": "依頼は『目的→期限→期待アウトプット』の順で短く。"
    },
    "Collaboration": {
        "ja_name": "協調性",
        "emoji": "🤝",
        "desc": "一緒に考える・すり合わせる・相手の意見を取り込む発言が多い軸です。",
        "scenes": ["認識合わせ", "合意形成", "複数案から選ぶとき"],
        "value": "対立を減らし、納得感のある合意をつくります。",
        "questions": ["合わせることが目的になっていませんか？", "本音と合意の距離はどれくらい？"],
        "high": "関係性が滑らかで、共通理解が作りやすい。",
        "low": "独自の主張が出にくく、判断が他者依存になりやすい。",
        "tips": "『賛成＋1点懸念＋1提案』の形で同調と前進を両立。"
    },
    "Active_Listening": {
        "ja_name": "傾聴性",
        "emoji": "🙋",
        "desc": "質問や確認を通じて相手の状況・意図を引き出す発言が多い軸です。",
        "scenes": ["状況把握", "問題の掘り下げ", "誤解を防ぎたいとき"],
        "value": "相手の文脈を理解し、ズレを減らします。",
        "questions": ["質問は“理解”のため？“誘導”のため？", "確認の頻度は相手に合っていますか？"],
        "high": "相手の背景が見えやすく、的外れを防げる。",
        "low": "情報が不足し、推測で進めて手戻りが起きやすい。",
        "tips": "深掘り→要約→確認（『つまり◯◯で合ってる？』）をセットに。"
    },
    "Logical_Expression": {
        "ja_name": "論理表出性",
        "emoji": "📖",
        "desc": "理由・根拠・構造（まず/次に/結論など）で筋道立てて説明する軸です。",
        "scenes": ["説明", "説得", "複雑な話の整理"],
        "value": "会話の透明性が上がり、理解が揃いやすい。",
        "questions": ["相手の前提（知識量）に合わせていますか？", "結論を先に置けていますか？"],
        "high": "分かりやすく誤解が減る。議論が収束しやすい。",
        "low": "話が散らばりやすく、意図が伝わりにくい。",
        "tips": "『結論→理由2つ→補足』の3段で短く。"
    },
    "Emotional_Expression": {
        "ja_name": "感情表出性",
        "emoji": "💬",
        "desc": "嬉しい/不安など感情や主観を言葉にする軸です（温度感が伝わります）。",
        "scenes": ["共感が欲しいとき", "温度差を埋めたいとき", "関係性を深めたいとき"],
        "value": "会話に人間味が出て、距離が縮まりやすい。",
        "questions": ["感情を言葉にするのは得意？苦手？", "感情の表現は相手にとって受け取りやすい？"],
        "high": "温度感が伝わり、関係性が強まりやすい。",
        "low": "冷たく見えたり、意図が誤解されやすい。",
        "tips": "『事実＋気持ち（短く）』で表現する。"
    },
    "Empathy_Care": {
        "ja_name": "配慮・共感性",
        "emoji": "🫶",
        "desc": "相手の負荷や気持ちに配慮し、感謝/謝罪/クッション言葉を使う軸です。",
        "scenes": ["依頼", "断り", "相手が疲れているとき"],
        "value": "安心感が生まれ、摩擦が減ります。",
        "questions": ["配慮が“遠慮”になっていませんか？", "言いにくいことほど、丁寧に言えていますか？"],
        "high": "関係性の安全性が上がり、話しやすい空気ができる。",
        "low": "強く/冷たく見え、衝突や誤解が起きやすい。",
        "tips": "依頼は『クッション→お願い→理由→感謝』の順で。"
    },
    "Brevity": {
        "ja_name": "簡潔性",
        "emoji": "✂️",
        "desc": "短く要点で伝える傾向を示す参考軸です（短いほど高くなる設計）。",
        "scenes": ["忙しい相手", "チャットでのタスク指示", "結論だけ欲しい場面"],
        "value": "相手の認知負荷を下げ、返しやすくします。",
        "questions": ["短さは“省略”になっていませんか？", "要点の前提は共有できていますか？"],
        "high": "読みやすくスピードが出る一方、情報不足になりやすい。",
        "low": "丁寧だが長文化しやすい（読む負担が増える）。",
        "tips": "1メッセージ1テーマ。箇条書き3点まで。"
    },
}

THINK_STYLE_GUIDE = {
    "Structural_Thinking": {
        "ja_name": "構造思考性",
        "emoji": "🗂️",
        "desc": "分類・枠組み・階層で物事を整理して捉える傾向です。",
        "scenes": ["全体像を掴みたい", "複雑な問題を分解", "説明を分かりやすくしたい"],
        "value": "見通しが立ち、議論が迷子になりにくい。",
        "questions": ["整理のため？説得のため？", "枠組みが現実を縛っていない？"],
        "high": "理解が早く、再現性が高い。",
        "low": "論点が散りやすく、共有が難しくなる。",
        "tips": "『結論→要素→優先順位』で枠を作る。"
    },
    "Abstractness": {
        "ja_name": "抽象度",
        "emoji": "🫧",
        "desc": "概念/本質/一般化の方向に思考が寄る傾向です（具体とのバランス）。",
        "scenes": ["原理原則を考える", "方針を決める", "他領域へ応用する"],
        "value": "本質を掴み、応用可能な学びに変換できます。",
        "questions": ["抽象→具体の往復ができていますか？", "相手の解像度に合わせていますか？"],
        "high": "俯瞰が効く反面、行動に落ちにくいことがある。",
        "low": "実務には強いが、方針や本質の議論が弱くなることがある。",
        "tips": "最後に『具体例を1つ』添える。"
    },
    "Multi_Perspective": {
        "ja_name": "多角性",
        "emoji": "👥",
        "desc": "別の観点・トレードオフ・反論など複数視点で考える傾向です。",
        "scenes": ["意思決定", "メリデメ比較", "リスクを見落としたくない"],
        "value": "盲点を減らし、納得感のある判断に近づきます。",
        "questions": ["視点を広げすぎて決められなくなっていませんか？", "最終的な判断軸は何ですか？"],
        "high": "バランスが良いが、決断が遅れやすい。",
        "low": "決断は早いが、盲点や反発が出やすい。",
        "tips": "『観点は3つまで→最後に判断軸で決める』。"
    },
    "Self_Reflection": {
        "ja_name": "内省性",
        "emoji": "🪞",
        "desc": "自分の状態・癖・学びを振り返って言語化する傾向です。",
        "scenes": ["改善したい", "迷いがある", "経験から学びたい"],
        "value": "成長速度が上がり、再発防止に繋がります。",
        "questions": ["内省が自己否定になっていませんか？", "次の一手に落とせていますか？"],
        "high": "学びが深いが、考えすぎで動けなくなることも。",
        "low": "行動は速いが、学びが蓄積しにくい。",
        "tips": "『気づき→原因→次の1アクション』の3点で締める。"
    },
    "Future_Oriented": {
        "ja_name": "未来志向性",
        "emoji": "🎯",
        "desc": "今後・計画・可能性に向けて思考が進む傾向です。",
        "scenes": ["ロードマップ", "目標設計", "次の打ち手を考える"],
        "value": "行動が前向きに繋がりやすい。",
        "questions": ["未来の話が現実逃避になっていませんか？", "次の1週間で何をする？"],
        "high": "前進力があるが、足元の詰めが甘くなることも。",
        "low": "堅実だが、変化や挑戦が起きにくい。",
        "tips": "『次の一手（期限つき）』まで落とす。"
    },
    "Risk_Awareness": {
        "ja_name": "リスク感知性",
        "emoji": "⚠️",
        "desc": "懸念・条件・失敗パターンを先読みする傾向です。",
        "scenes": ["公開/運用", "意思決定", "抜け漏れチェック"],
        "value": "安全装置になり、炎上や手戻りを減らします。",
        "questions": ["懸念提示の後に“対策”も出せていますか？", "リスクが不安を増幅していませんか？"],
        "high": "堅牢になるが、慎重すぎて前に進みにくいことも。",
        "low": "スピードは出るが、事故や手戻りが増えやすい。",
        "tips": "『懸念→影響→対策案』をセットで言う。"
    },
}


# CSS
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
.guide-card {
    background: #f8f9ff;
    border-left: 4px solid #667eea;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.1rem;
    margin-bottom: 1rem;
}
.guide-desc {
    margin: 0;
    font-size: 0.97rem;
    line-height: 1.7;
    color: #333;
}
.guide-question {
    background: #fff8e1;
    border-radius: 8px;
    padding: 0.65rem 0.9rem;
    margin-bottom: 0.6rem;
    font-size: 0.93rem;
    line-height: 1.6;
}
.question-num {
    font-weight: bold;
    color: #f9a825;
    margin-right: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ヘルパー関数
def process_files(file_pairs: List[tuple], my_name: str, min_chars: int, user_id: str) -> None:
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
                "user_id": user_id,
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
                # 新13軸スコアを計算（1メッセージ単位）
                from classify_rules import calculate_axis_scores
                scores = calculate_axis_scores([{"text": proc_text}])
                label_batch.append({"message_id": row_id, **scores})

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
        st.caption(f"ℹ️ パースできなかった行: 合計 {total_skip_lines} 行")


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


def render_compare_bar(cp_dist: Dict, global_dist: Dict, labels: List[str], cp_name: str) -> None:
    rows = []
    for label in labels:
        if label in COMM_STYLE_DISPLAY:
            display_label = COMM_STYLE_DISPLAY[label]
        elif label in THINK_STYLE_DISPLAY:
            display_label = THINK_STYLE_DISPLAY[label]
        else:
            display_label = label

        rows.append({"ラベル": display_label, "値": float(cp_dist.get(label, 0)), "種別": f"「{cp_name}」"})
        rows.append({"ラベル": display_label, "値": float(global_dist.get(label, 0)), "種別": "全体平均"})

    df = pd.DataFrame(rows)

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("ラベル:N", title=None),
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
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def render_radar_chart(dist_a: Dict, dist_b: Dict, labels: List[str], name_a: str, name_b: str) -> None:
    """
    レーダーチャートを表示（改善版）
    """
    values_a = [float(dist_a.get(l, 0)) for l in labels]
    values_b = [float(dist_b.get(l, 0)) for l in labels]
    
    # 閉じた図形にする
    values_a += [values_a[0]]
    values_b += [values_b[0]]
    labels_closed = labels + [labels[0]]
    
    # ラベルを改行して短くする（必要に応じて）
    labels_display = []
    for label in labels_closed:
        # 長いラベルは改行
        if len(label) > 6:
            # 適切な位置で改行
            if "・" in label:
                label = label.replace("・", "<br>")
            elif len(label) > 8:
                mid = len(label) // 2
                label = label[:mid] + "<br>" + label[mid:]
        labels_display.append(label)
    
    fig = go.Figure()
    
    # トレース1（相手）
    fig.add_trace(go.Scatterpolar(
        r=values_a,
        theta=labels_display,
        fill='toself',
        name=name_a,
        line=dict(color='#667eea', width=2),
        fillcolor='rgba(102, 126, 234, 0.3)',
    ))
    
    # トレース2（全体平均）
    fig.add_trace(go.Scatterpolar(
        r=values_b,
        theta=labels_display,
        fill='toself',
        name=name_b,
        line=dict(color='#cccccc', width=2),
        fillcolor='rgba(204, 204, 204, 0.2)',
    ))
    
    # レイアウト設定
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                # 目盛りを明示的に設定
                tickmode='linear',
                tick0=0,
                dtick=0.2,  # 0.2刻み（20%）
                tickformat='.0%',  # パーセント表示
                tickfont=dict(size=10),
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1,
            ),
            angularaxis=dict(
                # ラベル設定
                tickfont=dict(size=11),
                rotation=90,  # 回転
            ),
            bgcolor='rgba(255,255,255,0.9)',
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        height=500,  # 高さを拡大
        margin=dict(l=80, r=80, t=40, b=80),  # マージンを拡大
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_style_guide_card(key: str, data: dict) -> None:
    header = f"{data['emoji']} {data['ja_name']}  /  {key}"
    with st.expander(header, expanded=False):
        st.markdown(
            f'<div class="guide-card"><p class="guide-desc">{data["desc"]}</p></div>',
            unsafe_allow_html=True,
        )
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**よく現れる場面**")
            for scene in data["scenes"]:
                st.markdown(f"- {scene}")
            st.markdown("**このスタイルの価値**")
            st.markdown(data["value"])
        with col_right:
            st.markdown("**気づきのための問い**")
            for i, q in enumerate(data["questions"], 1):
                st.markdown(
                    f'<div class="guide-question"><span class="question-num">Q{i}</span>{q}</div>',
                    unsafe_allow_html=True,
                )

def render_top3_summary_for_guide(user_id: str) -> None:
    """スタイルガイド冒頭に表示する：あなたのTop3（全体=global）
    PCは2列、スマホは縦並びに自動対応。
    """
    messages = fetch_my_messages_with_labels(user_id)
    if not messages:
        st.info("データがないため、Top3は表示できません。まずは「取り込み」タブでLINEログを取り込んでください。")
        st.divider()
        return

    dr = build_distribution(messages)
    g = dr.get("global", {})
    sd = g.get("style_dist", {}) or {}
    td = g.get("think_dist", {}) or {}

    top_comm = sorted(sd.items(), key=lambda x: float(x[1]), reverse=True)[:3] if sd else []
    top_think = sorted(td.items(), key=lambda x: float(x[1]), reverse=True)[:3] if td else []

    def fmt_pct(v: float) -> str:
        return f"{float(v) * 100:.1f}%"

    # --- スマホ判定（CSSで幅を見てレイアウトを切り替え）---
    # StreamlitはPython側で確実な画面幅取得が難しいので、
    # 1) CSSでスマホ時は「2列」レイアウトを縦に
    # 2) さらにカード風で読みやすく
    st.markdown(
        """
        <style>
        /* Top3をカード風に（ダークモードでも文字が消えないように色を固定） */
        .top3-card {
            background: #f8f9ff;
            border: 1px solid #e8eaf6;
            border-radius: 14px;
            padding: 14px 16px;
            margin-bottom: 12px;

            /* ★重要：カード内の文字色を固定（ダークモード対策） */
            color: #111827; /* slate-900 */
        }
        .top3-title {
            font-size: 1.05rem;
            font-weight: 800;
            margin: 0 0 8px 0;

            color: #111827;
        }
        .top3-item {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            padding: 6px 0;
            border-bottom: 1px dashed #e8eaf6;
            font-size: 0.98rem;

            color: #111827;
        }
        .top3-item:last-child { border-bottom: none; }
        .top3-rank { font-weight: 800; color: #111827; }
        .top3-name { font-weight: 600; color: #111827; }
        .top3-val  { font-variant-numeric: tabular-nums; font-weight: 800; color: #111827; }

        /* スマホは余白を少し詰める */
        @media (max-width: 640px) {
            .top3-card { padding: 12px 12px; border-radius: 12px; }
            .top3-item { font-size: 0.95rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("あなたの全体傾向（Top3）")
    st.caption("※ 全トークルームを合算した傾向（global）です。単位：%")

    # ここはcolumnsを使うが、スマホでは自動的に縦積みになりやすい＋カードで視認性を確保
    col1, col2 = st.columns(2)

    def render_card(title: str, items: list, display_map: dict) -> None:
        if not items:
            body = '<div class="top3-item"><span class="top3-name">—</span><span class="top3-val"> </span></div>'
        else:
            rows = []
            for i, (k, v) in enumerate(items, 1):
                name = display_map.get(k, k)
                rows.append(
                    f'<div class="top3-item">'
                    f'<span class="top3-name"><span class="top3-rank">{i}.</span> {name}</span>'
                    f'<span class="top3-val">{fmt_pct(v)}</span>'
                    f'</div>'
                )
            body = "".join(rows)

        st.markdown(
            f'''
            <div class="top3-card">
              <div class="top3-title">{title}</div>
              {body}
            </div>
            ''',
            unsafe_allow_html=True,
        )

    with col1:
        render_card("🗣️ コミュニケーション Top3", top_comm, COMM_STYLE_DISPLAY)

    with col2:
        render_card("🧠 思考 Top3", top_think, THINK_STYLE_DISPLAY)

    st.divider()



def render_style_guide_tab() -> None:
    st.header("スタイルガイド")
    st.markdown("各スタイルの意味や役割を理解するためのガイドです。多い・少ないは良し悪しではなく、会話の中での **役割の違い** を示します。")
    st.divider()
    inner_tab_comm, inner_tab_think = st.tabs(["🗣️ コミュニケーションスタイル", "🧠 思考スタイル"])
    with inner_tab_comm:
        st.markdown("#### コミュニケーションスタイル（8種類）")
        st.caption("発言の「形」に現れる傾向です。相手との関わり方のパターンを示します。")
        st.markdown("")
        for key in COMM_STYLE_LABELS:
            if key in COMM_STYLE_GUIDE:
                render_style_guide_card(key, COMM_STYLE_GUIDE[key])
            else:
                st.warning(f"ガイド未定義: {key}")
    with inner_tab_think:
        st.markdown("#### 思考スタイル（6種類）")
        st.caption("発言の「中身」に現れる傾向です。何を重視して考えているかのパターンを示します。")
        st.markdown("")
        for key in THINK_STYLE_LABELS:
            if key in THINK_STYLE_GUIDE:
                render_style_guide_card(key, THINK_STYLE_GUIDE[key])
            else:
                st.warning(f"ガイド未定義: {key}")


# ページヘッダー
st.markdown('<div class="main-title">🪞 Prismee</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">相手ごとに、自分のコミュニケーション・思考スタイルの違いを可視化する</div>', unsafe_allow_html=True)
st.markdown('<div class="privacy-badge">🔒 生ログは外部LLMへ送信しません</div>', unsafe_allow_html=True)

st.info("ℹ️ **公開環境での利用について：** このアプリは公開サーバー上で動作します。入力したデータの取り扱いは利用者の責任で行ってください。")

# ✅ 招待リンク生成機能
with st.expander("🎫 招待リンクを発行する", expanded=False):
    st.markdown("### 新しい招待リンクを生成")
    st.warning(
        "⚠️ **重要な注意事項**\n\n"
        "生成されたリンクを他の人と共有すると、**あなたの分析結果や取り込んだデータがその人にも見えてしまいます。**\n\n"
        "- 自分専用の新しいスペースを作りたい場合のみ、このリンクを使ってください\n"
        "- 他の人に渡す場合は、必ず新しいリンクを別途生成してください\n"
        "- 各リンクは独立したパスコードで保護されます"
    )
    
    col_gen, col_info = st.columns([1, 2])
    with col_gen:
        if st.button("🔗 新しいリンクを生成", type="primary", use_container_width=True):
            base_url = "https://prismee.streamlit.app"
            new_uid = str(uuid.uuid4())
            new_link = f"{base_url}/?uid={new_uid}"
            st.session_state["generated_link"] = new_link
            st.session_state["generated_uid"] = new_uid
    
    with col_info:
        st.caption("💡 新しいスペースを作成したい場合や、他の人に独立した環境を提供したい場合に使用してください。")
    
    if "generated_link" in st.session_state:
        st.divider()
        st.success("✅ 招待リンクを生成しました")
        st.markdown("**📎 招待リンク：**")
        st.code(st.session_state["generated_link"], language=None)
        st.markdown("**🔑 UID：**")
        st.code(st.session_state["generated_uid"], language=None)
        st.info(
            "📌 このリンクにアクセスすると、初回は新しいパスコード設定が求められます。  \n"
            "📌 現在のスペース（このページ）とは完全に独立した別の空間になります。"
        )

st.divider()

# サイドバー
with st.sidebar:
    st.header("⚙️ 設定")
    st.caption(f"🔐 認証済み: {USER_ID[:8]}...")

    my_name = st.text_input("自分の表示名", value=st.session_state.get("my_name", ""), placeholder="例：太郎", help="LINEトーク履歴に表示される自分の名前")
    if my_name:
        st.session_state["my_name"] = my_name

    min_chars = st.slider("ノイズ除外・最小文字数", 1, 10, 2)

    st.divider()
    st.subheader("📊 DB 統計")
    stats = get_db_stats(USER_ID)
    st.metric("総メッセージ", stats["total_messages"])
    st.metric("自分のメッセージ", stats["my_messages"])
    st.metric("ラベル付き", stats["labeled_messages"])
    st.metric("ファイル数", stats["sources"])

    sources = fetch_sources(USER_ID)
    if sources:
        st.divider()
        st.subheader("🗑️ データ削除")
        del_src = st.selectbox("削除するソース", ["（選択してください）"] + sources, key="del_src_sel")
        if del_src != "（選択してください）":
            if st.button(f"「{del_src}」を削除", type="secondary"):
                cnt = delete_source(USER_ID, del_src)
                st.success(f"{cnt} 件削除しました")
                st.rerun()

# タブ
tab1, tab2, tab3, tab4 = st.tabs(["📥 取り込み", "📊 分析・可視化", "📤 LLM エクスポート", "📚 スタイルガイド"])

# Tab 1: 取り込み
with tab1:
    st.header("LINEトーク履歴の取り込み")
    if not my_name:
        st.warning("⚠️ まずサイドバーで「自分の表示名」を入力してください")
    else:
        st.info("**使い方：** LINEアプリ → トーク → メニュー → トーク履歴を送信 → `.txt` を保存  \n1ファイル = 1トークルームとして扱います。")
        uploaded_files = st.file_uploader("LINEトーク履歴 (.txt) をドラッグ＆ドロップ", type=["txt"], accept_multiple_files=True)
        col_a, col_b = st.columns(2)
        with col_a:
            if uploaded_files:
                if st.button("🚀 取り込む", type="primary", use_container_width=True):
                    pairs = [(f.read(), f.name) for f in uploaded_files]
                    process_files(pairs, my_name, min_chars, USER_ID)
                    st.rerun()
        with col_b:
            if st.button("🧪 サンプルデータで試す", use_container_width=True):
                sample_path = Path(__file__).parent / "sample_data" / "sample_line.txt"
                if sample_path.exists():
                    _name = my_name or "健悟"
                    if not my_name:
                        st.session_state["my_name"] = "健悟"
                    process_files([(sample_path.read_bytes(), "sample_line.txt")], _name, min_chars, USER_ID)
                    st.rerun()
                else:
                    st.error("sample_data/sample_line.txt が見つかりません")

# Tab 2: 分析・可視化
with tab2:
    st.header("コミュニケーション & 思考スタイル分析")
    messages = fetch_my_messages_with_labels(USER_ID)
    if not messages:
        st.info("データがありません。「取り込み」タブで LINEログを取り込んでください。")
    else:
        dist_result = build_distribution(messages)
        diffs_all = calc_diff_from_global(dist_result)
        df_style, df_think = dist_to_dataframe(dist_result)
        g = dist_result.get("global", {})
        counterparties = [cp for cp in dist_result.keys() if cp != "global"]

        st.subheader("📈 全体サマリー")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("分析メッセージ数", g.get("count", 0))
        with c2:
            st.metric("トークルーム数", len(counterparties))
        with c3:
            sd = g.get("style_dist", {})
            ts_key = max(sd, key=sd.get) if sd else None
            ts = COMM_STYLE_DISPLAY.get(ts_key, "—") if ts_key else "—"
            st.metric("最頻コミュスタイル", ts, f"{sd.get(ts_key, 0):.0%}" if ts_key else "—")
        with c4:
            td = g.get("think_dist", {})
            tt_key = max(td, key=td.get) if td else None
            tt = THINK_STYLE_DISPLAY.get(tt_key, "—") if tt_key else "—"
            st.metric("最頻思考スタイル", tt, f"{td.get(tt_key, 0):.0%}" if tt_key else "—")

        st.subheader("🎨 コミュニケーションスタイル分布")
        render_grouped_bar(df_style.rename(columns=COMM_STYLE_DISPLAY), [COMM_STYLE_DISPLAY[k] for k in COMM_STYLE_LABELS])
        
        st.subheader("🧠 思考スタイル分布")
        render_grouped_bar(df_think.rename(columns=THINK_STYLE_DISPLAY), [THINK_STYLE_DISPLAY[k] for k in THINK_STYLE_LABELS])

        st.divider()
        st.subheader("👤 相手別スタイル詳細")
        if not counterparties:
            st.info("相手別データがありません")
        else:
            sel = st.selectbox("分析するトークルームを選択", ["（全相手を比較）"] + counterparties)
            if sel == "（全相手を比較）":
                tab_cs, tab_ts = st.tabs(["コミュニケーションスタイル", "思考スタイル"])

                def render_table_with_global_fixed(df: pd.DataFrame, labels: List[str], rename_map: Dict[str, str]):
                    table = df[labels].rename(columns=rename_map).copy()

                    # ✅ ここで 100倍（表示用）
                    table = table * 100

                    global_row = table.loc[["global"]] if "global" in table.index else None
                    others = table.drop(index=["global"], errors="ignore")

                    if global_row is not None:
                        st.caption("📌 global（全体平均）は固定表示")
                        st.dataframe(
                            global_row,
                            use_container_width=True,
                            hide_index=False,
                            column_config={
                                col: st.column_config.NumberColumn(format="%.1f")
                                for col in global_row.columns
                            },
                        )
                        st.markdown("")

                    st.caption("⬇️ 以降はクリックで数値ソートできます（単位：%）")
                    st.dataframe(
                        others,
                        use_container_width=True,
                        hide_index=False,
                        column_config={
                            col: st.column_config.NumberColumn(format="%.1f")
                            for col in others.columns
                        },
                    )


                with tab_cs:
                    render_table_with_global_fixed(df_style, COMM_STYLE_LABELS, COMM_STYLE_DISPLAY)

                with tab_ts:
                    render_table_with_global_fixed(df_think, THINK_STYLE_LABELS, THINK_STYLE_DISPLAY)


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
                        dv = float(item.get("diff", 0))
                        sign = "▲" if dv > 0 else "▼"
                        color = "#2563eb" if dv > 0 else "#dc2626"
                        
                        # ✅ display フィールドを使う（既に日本語化済み）
                        display_name = item.get("display", item.get("label", ""))
                        
                        # kind も日本語化
                        kind_raw = item.get("kind", "")
                        if kind_raw == "comm":
                            kind_label = "コミュニケーション"
                        elif kind_raw == "think":
                            kind_label = "思考"
                        else:
                            kind_label = kind_raw
                        with cols3[i]:
                            html = f"""<div style="background:#f8f9ff;border:1px solid #e8eaf6;border-radius:12px;padding:16px 18px;text-align:center;">
                            <div style="font-size:0.75rem;color:#888;">{kind_label}スタイル</div>
                            <div style="font-size:1.3rem;font-weight:700;margin:6px 0;">{display_name}</div>
                            <div style="font-size:1.05rem;color:{color};font-weight:700;">{sign}{abs(dv):.1%}</div></div>"""
                            components.html(html, height=120)

                st.markdown("#### 🕸️ レーダーチャート比較（相手 vs 全体平均）")
                st.caption("💡 外側に行くほど高スコア｜青=この相手、灰=全体平均")

                col_r1, col_r2 = st.columns(2)

                with col_r1:
                    st.markdown("**📊 コミュニケーションスタイル**")
                    comm_labels_disp = [COMM_STYLE_DISPLAY[k] for k in COMM_STYLE_LABELS]
                    cp_comm = {COMM_STYLE_DISPLAY[k]: float(cp_data.get("style_dist", {}).get(k, 0)) for k in COMM_STYLE_LABELS}
                    g_comm = {COMM_STYLE_DISPLAY[k]: float(g_data.get("style_dist", {}).get(k, 0)) for k in COMM_STYLE_LABELS}
                    render_radar_chart(cp_comm, g_comm, comm_labels_disp, f"「{sel}」", "全体平均")

                with col_r2:
                    st.markdown("**🧠 思考スタイル**")
                    think_labels_disp = [THINK_STYLE_DISPLAY[k] for k in THINK_STYLE_LABELS]
                    cp_think = {THINK_STYLE_DISPLAY[k]: float(cp_data.get("think_dist", {}).get(k, 0)) for k in THINK_STYLE_LABELS}
                    g_think = {THINK_STYLE_DISPLAY[k]: float(g_data.get("think_dist", {}).get(k, 0)) for k in THINK_STYLE_LABELS}
                    render_radar_chart(cp_think, g_think, think_labels_disp, f"「{sel}」", "全体平均")
                st.markdown("#### コミュニケーションスタイル 比較")
                render_compare_bar(cp_data.get("style_dist", {}), g_data.get("style_dist", {}), COMM_STYLE_LABELS, sel)
                st.markdown("#### 思考スタイル 比較")
                render_compare_bar(cp_data.get("think_dist", {}), g_data.get("think_dist", {}), THINK_STYLE_LABELS, sel)

# Tab 3: LLM エクスポート
with tab3:
    st.header("外部 LLM へのエクスポート（任意）")
    st.markdown("**🔒 プライバシー：** 生メッセージは含まれません（集計比率データのみ）  \n**使い方：** JSON を生成 → コピー → ChatGPT/Claude に「プロンプト」と共に貼り付け")
    _my_name = st.session_state.get("my_name", "ユーザー")
    msgs_exp = fetch_my_messages_with_labels(USER_ID)
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
            st.text_area("集計 JSON（コピーして LLM に貼り付けてください）", value=st.session_state["summary_json"], height=380)
            st.download_button("⬇️ JSON をダウンロード", data=st.session_state["summary_json"].encode("utf-8"), file_name="zenbu_jibun_summary.json", mime="application/json")
            st.divider()
            st.subheader("📝 インサイト生成プロンプト")
            prompt_path = Path(__file__).parent / "prompts" / "insight_prompt.txt"
            if prompt_path.exists():
                st.text_area("このプロンプトの後に上記 JSON を貼り付けて LLM に送信してください", value=prompt_path.read_text(encoding="utf-8"), height=300)
            else:
                st.warning("prompts/insight_prompt.txt が見つかりません")

# Tab 4: スタイルガイド
with tab4:
    render_top3_summary_for_guide(USER_ID)
    render_style_guide_tab()
