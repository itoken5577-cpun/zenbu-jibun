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
    "Ask": {
        "ja_name": "問いかけ",
        "emoji": "🙋",
        "desc": "相手に質問を投げかけたり、確認を取ったりする発言が多いスタイルです。「どう思う？」「教えてもらえる？」のように、会話の中で相手の考えや状況を引き出そうとする動きが特徴です。",
        "scenes": [
            "相手の意見や気持ちを確認したいとき",
            "状況がよくわからず、情報を集めたいとき",
            "相手に決断や選択を促す前に、認識を合わせたいとき",
        ],
        "value": "会話に「余白」をつくり、相手が話しやすい空気を生み出します。一方的になりがちな対話を双方向に保ち、相手が「聞いてもらえている」と感じやすくなります。",
        "questions": [
            "あなたが質問するとき、それは「知りたい」からですか、それとも「確認したい」からですか？",
            "質問の多い相手との会話と少ない相手との会話で、何か違いを感じますか？",
        ],
    },
    "Propose": {
        "ja_name": "提案",
        "emoji": "💡",
        "desc": "アイデアや選択肢を積極的に提示する発言が多いスタイルです。「こんな方法はどう？」「試してみたらいいかも」のように、会話に新しい方向性を差し込む動きが特徴です。",
        "scenes": [
            "話し合いが行き詰まっているとき",
            "相手が悩んでいて、何か手がかりを求めているとき",
            "自分の中にアイデアが浮かんで、共有したくなったとき",
        ],
        "value": "会話に「動き」をもたらし、停滞を打破するきっかけになります。相手に押しつけるのではなく「選択肢を増やす」ことで、二人の間に新しい可能性を開くことができます。",
        "questions": [
            "提案するとき、相手の反応についてどんなことを気にしていますか？",
            "「提案しすぎた」と感じた経験はありますか？そのときどんな状況でしたか？",
        ],
    },
    "Structure": {
        "ja_name": "整理",
        "emoji": "🗂️",
        "desc": "情報を箇条書きにしたり、話の順番を整えたりする発言が多いスタイルです。「まず〜、次に〜、最後に〜」のように、複雑な内容をわかりやすい形に組み立てようとする動きが特徴です。",
        "scenes": [
            "複数の話題をまとめて伝えたいとき",
            "相手が混乱していて、情報を整理してあげたいとき",
            "計画や段取りを共有するとき",
        ],
        "value": "会話に「見通し」を与え、相手が全体像を把握しやすくなります。とくに情報量が多い場面で、やりとりがスムーズになる土台をつくります。",
        "questions": [
            "整理して話すとき、それは自分のためですか、相手のためですか？",
            "整理されすぎた会話の中で、何か窮屈さを感じることはありますか？",
        ],
    },
    "Empathize": {
        "ja_name": "共感",
        "emoji": "🤝",
        "desc": "相手の感情や状況に寄り添う発言が多いスタイルです。「大変だったね」「それはうれしいね」のように、言葉で気持ちを受け取ろうとする動きが特徴です。",
        "scenes": [
            "相手が何か辛いことや嬉しいことを話しているとき",
            "相手が解決策より「聞いてほしい」状態にあるとき",
            "場の雰囲気を温めたいとき",
        ],
        "value": "相手に「受け取ってもらえた」という感覚を届けます。問題解決よりも先に感情的なつながりをつくることで、その後の会話が深まりやすくなります。",
        "questions": [
            "共感を示すとき、自分自身もその感情を感じていますか？",
            "共感したいのに言葉が出てこないと感じる場面はありますか？",
        ],
    },
    "Explain": {
        "ja_name": "説明",
        "emoji": "📖",
        "desc": "物事の背景・理由・仕組みを丁寧に言語化する発言が多いスタイルです。「なぜなら〜」「つまり〜ということで」のように、理解を助けようとする動きが特徴です。",
        "scenes": [
            "相手に状況や意図を正確に伝えたいとき",
            "自分の判断や行動の根拠を共有したいとき",
            "誤解が生まれそうな場面で丁寧にフォローしたいとき",
        ],
        "value": "会話に「透明性」をもたらします。なぜそう言ったのか、なぜそう動いたのかが伝わることで、相手との認識のずれが生まれにくくなります。",
        "questions": [
            "説明が多くなるとき、その根底にどんな気持ちがありますか？",
            "「説明しすぎたかも」と感じた会話を振り返ると、何が見えてきますか？",
        ],
    },
    "Lead": {
        "ja_name": "主導",
        "emoji": "🧭",
        "desc": "方向性を示したり、依頼・指示・決定を伝える発言が多いスタイルです。「これをお願いします」「確定にしましょう」のように、会話に推進力をもたらす動きが特徴です。",
        "scenes": [
            "物事を前に進める必要があるとき",
            "誰かが動き出すきっかけを作りたいとき",
            "期限やタスクを明確にしたいとき",
        ],
        "value": "会話を「行動」につなげる役割を果たします。話し合いが続いているだけで何も決まらない状況に、具体的な一歩を生み出すことができます。",
        "questions": [
            "主導的に動くとき、それは自分がやりたいからですか、誰かがやらないからですか？",
            "主導する役割を誰かに渡したいと感じる瞬間はありますか？",
        ],
    },
    "Align": {
        "ja_name": "同調",
        "emoji": "🔗",
        "desc": "相手の意見や提案に応じたり、場の流れに合わせたりする発言が多いスタイルです。「たしかに」「了解」「いいと思う」のように、摩擦を減らしながら場を整える動きが特徴です。",
        "scenes": [
            "相手の考えに素直に共鳴したとき",
            "対立より協調を大切にしたいとき",
            "決定事項を受け入れ、次のステップに進みたいとき",
        ],
        "value": "会話の「なめらかさ」を生み出します。意見のぶつかりを和らげ、全体の関係性を保ちながら物事を進める潤滑油のような役割を果たします。",
        "questions": [
            "同調するとき、自分の本音も一緒にそこにありますか？",
            "「本当は違う意見があったけど合わせた」という経験から、何を感じましたか？",
        ],
    },
    "SmallTalk": {
        "ja_name": "雑談",
        "emoji": "💬",
        "desc": "挨拶・日常のやりとり・軽いユーモアなど、目的を持たない会話が多いスタイルです。「おはよう」「最近どう？」のように、関係をほぐすための言葉が特徴です。",
        "scenes": [
            "会話の始まりや終わりのとき",
            "久しぶりに連絡を取るとき",
            "重い話題の後に場を和らげたいとき",
        ],
        "value": "会話に「人間らしい温かさ」を加えます。用件だけのやりとりでは生まれにくい、ゆるやかな信頼感やつながりの感覚を育てる役割があります。",
        "questions": [
            "雑談が多くなる相手とそうでない相手の違いは、どこにあると思いますか？",
            "雑談をするとき、自分の中にどんな気持ちがありますか？",
        ],
    },
}

THINK_STYLE_GUIDE = {
    "Logic": {
        "ja_name": "論理",
        "emoji": "⚙️",
        "desc": "理由・根拠・因果関係を言語化する発言が多いスタイルです。「なぜなら〜」「データによると〜」のように、物事を筋道立てて説明しようとする動きが特徴です。",
        "scenes": [
            "判断の根拠を相手に伝えたいとき",
            "数字やデータを使って説明するとき",
            "複雑な問題を整理して考えるとき",
        ],
        "value": "会話に「説得力」と「一貫性」をもたらします。感情や直感だけでなく、論拠に基づいて話すことで、相手との認識合わせがしやすくなります。",
        "questions": [
            "論理的に話すとき、相手にどんな状態になってほしいと思っていますか？",
            "根拠を求めたくなる場面と、直感を優先したい場面の違いは何ですか？",
        ],
    },
    "Other": {
        "ja_name": "他者視点",
        "emoji": "👥",
        "desc": "相手や周囲の立場・気持ちを意識した発言が多いスタイルです。「相手はどう感じるか」「みんなにとってどうか」のように、自分以外の視点から考えようとする動きが特徴です。",
        "scenes": [
            "チームや周囲への影響を考えるとき",
            "相手の立場に立って言葉を選ぶとき",
            "自分の意見より関係性を優先したいとき",
        ],
        "value": "会話に「配慮の深さ」をもたらします。自分の考えだけでなく相手の文脈を取り込もうとする姿勢が、信頼感のある対話を生み出します。",
        "questions": [
            "他者の視点を意識するとき、自分の意見はどこにありますか？",
            "「相手のことを考えすぎた」と感じた経験はありますか？",
        ],
    },
    "Goal": {
        "ja_name": "目標志向",
        "emoji": "🎯",
        "desc": "目的・成果・ゴールを意識した発言が多いスタイルです。「何のためにやるのか」「どこを目指しているのか」のように、会話を目標に結びつけようとする動きが特徴です。",
        "scenes": [
            "プロジェクトの方向性を確認するとき",
            "優先順位を決めたいとき",
            "成果や達成基準を共有するとき",
        ],
        "value": "会話に「方向感」をもたらします。何のために話しているのかが明確になることで、議論が散漫になるのを防ぎ、行動につながりやすくなります。",
        "questions": [
            "ゴールを意識するとき、それは自分のゴールですか、相手や組織のゴールですか？",
            "目標のない雑談や探索的な会話の中で、どんな気持ちになりますか？",
        ],
    },
    "Risk": {
        "ja_name": "リスク察知",
        "emoji": "⚠️",
        "desc": "問題・懸念・リスクを先読みした発言が多いスタイルです。「このままだと〜になりかねない」「万が一のとき〜」のように、起こりうる課題を事前に言語化しようとする動きが特徴です。",
        "scenes": [
            "計画に抜け漏れがないか確認するとき",
            "相手の判断に危うさを感じたとき",
            "物事が動き出す前に注意点を伝えたいとき",
        ],
        "value": "会話に「安全装置」をもたらします。問題を後から発見するのではなく、事前に可視化することで、チームや相手が安心して進める土台をつくります。",
        "questions": [
            "リスクを口にするとき、その背景にどんな気持ちがありますか？",
            "懸念を伝えたことで、会話の流れが変わった経験はありますか？",
        ],
    },
    "Explore": {
        "ja_name": "探索",
        "emoji": "🔭",
        "desc": "新しいアイデアや可能性を探ろうとする発言が多いスタイルです。「面白そう」「試してみたい」「どうなるんだろう」のように、未知のものに向かって思考を広げようとする動きが特徴です。",
        "scenes": [
            "新しいツールや方法に興味を持ったとき",
            "現状に対してもっと良い方法があるかもしれないと感じるとき",
            "学びや発見を誰かに共有したくなったとき",
        ],
        "value": "会話に「広がり」と「可能性」をもたらします。既存の枠にとらわれない問いや視点が、相手の思考にも新しい刺激を与えることがあります。",
        "questions": [
            "探索したくなるとき、何がそのスイッチになっていますか？",
            "アイデアを広げたいときと、収束させたいときの切り替えをどうしていますか？",
        ],
    },
    "Stability": {
        "ja_name": "安定志向",
        "emoji": "⚓",
        "desc": "継続・維持・確実性を意識した発言が多いスタイルです。「いつも通りで」「実績があるから」「着実に進めよう」のように、変化よりも安定を大切にしようとする動きが特徴です。",
        "scenes": [
            "実績のある方法を継続したいとき",
            "急な変化に慎重になりたいとき",
            "チームのペースや習慣を守りたいとき",
        ],
        "value": "会話に「信頼感」と「持続性」をもたらします。変化が多い環境の中で、一貫したスタンスを保つことが、周囲に安心感を与えることがあります。",
        "questions": [
            "安定を大切にするとき、何を守ろうとしていますか？",
            "変化を求める声と安定を求める自分の間で、どう折り合いをつけていますか？",
        ],
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


def render_style_guide_tab() -> None:
    st.header("スタイルガイド")
    st.markdown("各スタイルの意味や役割を理解するためのガイドです。多い・少ないは良し悪しではなく、会話の中での **役割の違い** を示します。")
    st.divider()
    inner_tab_comm, inner_tab_think = st.tabs(["🗣️ コミュニケーションスタイル", "🧠 思考スタイル"])
    with inner_tab_comm:
        st.markdown("#### コミュニケーションスタイル（8種類）")
        st.caption("発言の「形」に現れる傾向です。相手との関わり方のパターンを示します。")
        st.markdown("")
        for key, data in COMM_STYLE_GUIDE.items():
            render_style_guide_card(key, data)
    with inner_tab_think:
        st.markdown("#### 思考スタイル（6種類）")
        st.caption("発言の「中身」に現れる傾向です。何を重視して考えているかのパターンを示します。")
        st.markdown("")
        for key, data in THINK_STYLE_GUIDE.items():
            render_style_guide_card(key, data)


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

                    # global 行とそれ以外に分割
                    global_row = table.loc[["global"]] if "global" in table.index else None
                    others = table.drop(index=["global"], errors="ignore")

                    # ✅ 固定表示：global（1行だけ）
                    if global_row is not None:
                        st.caption("📌 global（全体平均）は固定表示")
                        st.dataframe(
                            global_row,
                            use_container_width=True,
                            hide_index=False,
                            column_config={
                                col: st.column_config.NumberColumn(format="%.1f%%")
                                for col in global_row.columns
                            },
                        )
                        st.markdown("")

                    # ✅ ソート可能：残り
                    st.caption("⬇️ 以降はクリックで数値ソートできます")
                    st.dataframe(
                        others,
                        use_container_width=True,
                        hide_index=False,
                        column_config={
                            col: st.column_config.NumberColumn(format="%.1f%%")
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
    render_style_guide_tab()