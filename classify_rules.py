"""
classify_rules.py - 13軸スコアリング（詳細マーカー辞書版）
"""
from typing import Dict, List
from collections import defaultdict
import re


# ─────────────────────────────────────
# 13軸の定義
# ─────────────────────────────────────

# コミュニケーション軸（7軸）
COMM_STYLE_LABELS = [
    "Lead_Directiveness",      # 主導性
    "Collaboration",           # 協調性
    "Active_Listening",        # 傾聴性
    "Logical_Expression",      # 論理表出性
    "Emotional_Expression",    # 感情表出性
    "Empathy_Care",           # 配慮・共感性
    "Brevity",                # 簡潔性（参考）
]

COMM_STYLE_DISPLAY = {
    "Lead_Directiveness": "主導性",
    "Collaboration": "協調性",
    "Active_Listening": "傾聴性",
    "Logical_Expression": "論理表出性",
    "Emotional_Expression": "感情表出性",
    "Empathy_Care": "配慮・共感性",
    "Brevity": "簡潔性",
}

# 思考軸（6軸）
THINK_STYLE_LABELS = [
    "Structural_Thinking",     # 構造思考性
    "Abstractness",           # 抽象度
    "Multi_Perspective",      # 多角性
    "Self_Reflection",        # 内省性
    "Future_Oriented",        # 未来志向性
    "Risk_Awareness",         # リスク感知性
]

THINK_STYLE_DISPLAY = {
    "Structural_Thinking": "構造思考性",
    "Abstractness": "抽象度",
    "Multi_Perspective": "多角性",
    "Self_Reflection": "内省性",
    "Future_Oriented": "未来志向性",
    "Risk_Awareness": "リスク感知性",
}


# ─────────────────────────────────────
# 詳細マーカー辞書
# ─────────────────────────────────────

MARKERS = {
    # ===== 1. 主導性 =====
    "directive": ["して", "してください", "しよう", "やりましょう", "行きます", "やって"],
    "assertive": ["決める", "決定", "確定", "〜だ", "〜である"],
    "proposal": ["結論", "方針", "次は", "まず", "方針は"],
    
    # ===== 2. 協調性 =====
    "collaborative": ["一緒に", "合わせて", "すり合わせ", "合意", "認識合わせ", "協力"],
    "options": ["もあり", "選択肢", "どちらか", "案"],
    "seek_opinion": ["どう思う", "意見", "考え", "どうでしょう", "どうかな"],
    
    # ===== 3. 傾聴性 =====
    "question_deep": ["なぜ", "どうやって", "具体的には", "背景は", "前提は", "理由は"],
    "question_clarify": ["教えて", "詳しく", "もう少し", "意図は", "確認", "聞きたい"],
    "question_emotion": ["困ってる", "大変", "どう感じ", "気持ち"],
    
    # ===== 4. 論理表出性 =====
    "structure": ["まず", "次に", "つまり", "結論", "根拠", "前提", "整理"],
    "causal": ["なので", "したがって", "なぜなら", "だから", "故に", "ため"],
    "analytical": ["要点", "因果", "仮説", "検証", "分析", "論理"],
    
    # ===== 5. 感情表出性 =====
    "emotion_positive": ["嬉しい", "楽しい", "好き", "ワクワク", "最高", "幸せ", "よかった"],
    "emotion_negative": ["不安", "つらい", "悲しい", "ムカつく", "焦る", "モヤモヤ", "困る"],
    "emotion_neutral": ["感じる", "思う", "気持ち"],
    "subjective": ["個人的に", "私は", "正直", "自分的に"],
    
    # ===== 6. 配慮・共感性 =====
    "empathy": ["わかる", "なるほど", "たしかに", "それな", "同意", "うんうん"],
    "care": ["お気持ち", "大変", "無理ない", "助かる", "気をつけて", "無理しないで"],
    "thanks": ["ありがとう", "ありがと", "感謝", "サンキュー", "ありです"],
    "apology": ["すみません", "ごめん", "申し訳", "失礼", "すまん"],
    "cushion": ["もしよければ", "差し支えなければ", "恐縮", "お手数", "できれば"],
    
    # ===== 7. 簡潔性（参考） =====
    "digression": ["ちなみに", "余談", "話はそれます"],
    "connector": ["あと", "それと", "ついでに"],
       
    # ===== 8. 構造思考性 =====
    "classification": [
        "分類", "要素", "観点", "軸", "カテゴリ", "種類",
        # ✅ 追加：より一般的な言葉
        "パターン", "タイプ", "グループ", "分けると", "2つ", "3つ",
    ],
    "framework": [
        "KPI", "KGI", "フレーム", "ポイント", "体系",
        # ✅ 追加
        "整理すると", "まとめると", "分けて", "ステップ",
    ],
    "hierarchy": [
        "大枠", "詳細", "上位", "下位", "中でも", "階層",
        # ✅ 追加
        "全体", "部分", "メイン", "サブ",
    ],
    
    # ===== 10. 多角性 =====
    "perspective": [
        "一方で", "別の観点", "逆に", "他方", "視点", "見方",
        # ✅ 追加
        "別の", "もう一つ", "他の", "反対",
    ],
    "multiple_options": [
        "メリデメ", "メリット", "デメリット", "トレードオフ", "両面",
        # ✅ 追加
        "良い点", "悪い点", "利点", "欠点", "両方",
    ],
    "anticipate": [
        "という見方", "ただし", "反論", "懸念", "考慮",
        # ✅ 追加
        "でも", "しかし", "とはいえ", "ただ",
    ],
    
    # ===== 11. 内省性 =====
    "self_reference": [
        "私は", "自分は", "自分としては", "個人的", "僕は", "俺は",
        # ✅ 追加（これは既に十分かも）
    ],
    "mental_process": [
        "気づいた", "感じた", "迷う", "大事にしたい", "思った",
        # ✅ 追加
        "考えた", "悩む", "迷ってる", "わからない",
    ],
    "self_improvement": [
        "苦手", "課題", "改善したい", "伸ばしたい", "振り返る", "反省",
        # ✅ 追加
        "直したい", "変えたい", "成長", "学び",
    ],
    
    # ===== 12. 未来志向性 =====
    "future_time": [
        "今後", "将来", "これから", "次に", "中長期", "先",
        # ✅ 追加
        "明日", "来週", "来月", "来年", "後で", "いつか",
    ],
    "planning": [
        "目指す", "ロードマップ", "スケジュール", "プラン", "ゴール", "計画",
        # ✅ 追加
        "予定", "準備", "段取り", "やること",
    ],
    "possibility": [
        "できそう", "なりうる", "可能性", "想定", "見込み",
        # ✅ 追加
        "かも", "できる", "なるかも", "いけそう",
    ],
    
    # ===== 13. リスク感知性 =====
    "risk": [
        "リスク", "懸念", "問題", "炎上", "批判", "副作用", "危険",
        # ✅ 追加
        "心配", "不安", "怖い", "まずい", "ヤバい",
    ],
    "conditional": [
        "ただし", "場合によって", "依存する", "条件", "もし",
        # ✅ 追加
        "次第", "によって", "なら", "だったら",
    ],
    "failure": [
        "最悪", "詰む", "破綻", "漏洩", "失敗", "ダメ",
        # ✅ 追加
        "無理", "できない", "厳しい", "間に合わない",
    ],
}


# ─────────────────────────────────────
# 特徴量抽出
# ─────────────────────────────────────

def count_emoji(text: str) -> int:
    """絵文字の数をカウント（簡易版）"""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # 顔文字
        "\U0001F300-\U0001F5FF"  # 記号
        "\U0001F680-\U0001F6FF"  # 交通
        "\U0001F1E0-\U0001F1FF"  # 旗
        "]+",
        flags=re.UNICODE
    )
    return len(emoji_pattern.findall(text))


def extract_features(text: str) -> dict:
    """
    テキストから詳細な特徴量を抽出
    """
    text_lower = text.lower()
    
    features = {
        "char_count": len(text),
        "sentence_count": max(1, text.count("。") + text.count(".") + text.count("\n")),
    }
    
    # 質問マーク
    features["question_mark"] = text.count("?") + text.count("？")
    
    # 感嘆符（cap付き）
    features["exclaim"] = min(text.count("!") + text.count("！"), 3)
    
    # 絵文字（cap付き）
    features["emoji_count"] = min(count_emoji(text), 3)
    
    # 笑い記号（cap付き）
    laugh_count = text.lower().count("笑") + text.lower().count("w")
    features["laugh"] = min(laugh_count, 3)
    
    # 箇条書きマーカー
    list_markers = ["1.", "2.", "3.", "①", "②", "③", "- ", "・"]
    features["list_marker"] = sum(1 for m in list_markers if m in text)
    
    # 各マーカーカテゴリのカウント
    for category, words in MARKERS.items():
        count = sum(1 for word in words if word in text_lower)
        features[f"marker_{category}"] = count
    
    # 敬語率（補助指標）
    polite_patterns = ["です", "ます", "ください", "いたします"]
    polite_count = sum(text.count(p) for p in polite_patterns)
    features["polite_count"] = polite_count
    
    # 数字の出現（具体性の指標）
    features["number_count"] = len(re.findall(r'\d+', text))
    
    return features


# ─────────────────────────────────────
# スコア計算
# ─────────────────────────────────────

def calculate_axis_scores(messages: list) -> dict:
    """
    メッセージリストから13軸のスコア（割合）を計算
    
    戻り値: {
        "Lead_Directiveness": 0.23,  # 23%
        ...
    }
    """
    if not messages:
        return {axis: 0.0 for axis in COMM_STYLE_LABELS + THINK_STYLE_LABELS}
    
    total_count = len(messages)
    total_chars = sum(len(m.get("text", "")) for m in messages)
    
    # 全メッセージの特徴量を集計
    features_sum = defaultdict(float)
    for msg in messages:
        feats = extract_features(msg.get("text", ""))
        for key, val in feats.items():
            features_sum[key] += val
    
    scores = {}
    
    # ===== 1. 主導性 =====
    directive = features_sum["marker_directive"] + features_sum["marker_assertive"]
    directive += features_sum["marker_proposal"] * 1.5
    # 質問が多いと下がる
    directive -= features_sum["question_mark"] * 0.3
    scores["Lead_Directiveness"] = max(0, directive / total_count)
    
    # ===== 2. 協調性 =====
    collab = features_sum["marker_collaborative"] * 1.5
    collab += features_sum["marker_options"] + features_sum["marker_seek_opinion"]
    scores["Collaboration"] = collab / total_count
    
    # ===== 3. 傾聴性 =====
    listening = features_sum["question_mark"] * 1.2
    listening += features_sum["marker_question_deep"] * 2
    listening += features_sum["marker_question_clarify"] * 1.5
    listening += features_sum["marker_question_emotion"]
    scores["Active_Listening"] = listening / total_count
    
    # ===== 4. 論理表出性 =====
    logical = features_sum["marker_structure"] * 1.5
    logical += features_sum["marker_causal"] * 1.2
    logical += features_sum["marker_analytical"]
    logical += features_sum["list_marker"] * 0.5
    scores["Logical_Expression"] = logical / total_count
    
    # ===== 5. 感情表出性 =====
    emotional = features_sum["marker_emotion_positive"]
    emotional += features_sum["marker_emotion_negative"]
    emotional += features_sum["marker_emotion_neutral"] * 0.8
    emotional += features_sum["marker_subjective"] * 0.5
    # 記号は cap 適用済み、低ウェイト
    emotional += (features_sum["exclaim"] + features_sum["emoji_count"] + features_sum["laugh"]) * 0.2
    scores["Emotional_Expression"] = emotional / total_count
    
    # ===== 6. 配慮・共感性 =====
    empathy = features_sum["marker_empathy"] * 1.5
    empathy += features_sum["marker_care"] * 1.2
    empathy += features_sum["marker_thanks"] * 1.2
    empathy += features_sum["marker_apology"]
    empathy += features_sum["marker_cushion"] * 1.3
    # 敬語は補助（低ウェイト）
    empathy += features_sum["polite_count"] * 0.05
    scores["Empathy_Care"] = empathy / total_count
    
    # ===== 7. 簡潔性（参考） =====
    avg_chars = total_chars / total_count if total_count > 0 else 0
    # 100文字を基準に正規化（短いほど高い）
    brevity = 1.0 - min(avg_chars / 100, 1.0)
    # 余談マーカーがあると下がる
    brevity -= features_sum["marker_digression"] / total_count * 0.2
    scores["Brevity"] = max(0, brevity)
    
    # ===== 8. 構造思考性 =====
    structural = features_sum["marker_classification"] * 1.5
    structural += features_sum["marker_framework"] * 1.3
    structural += features_sum["marker_hierarchy"]
    scores["Structural_Thinking"] = structural / total_count
    
    # ===== 9. 抽象度（差分） =====
    abstract = features_sum["marker_abstract"]
    concrete = features_sum["marker_concrete"] + features_sum["number_count"] * 0.2

    # ✅ 修正：マーカーがない場合は低スコアにする
    if abstract == 0 and concrete == 0:
        # どちらもない場合は中立（具体的でも抽象的でもない）
        scores["Abstractness"] = 0.1  # 低めに設定
    else:
        # 差分を計算
        abstractness_raw = (abstract - concrete) / total_count
        # 0〜1の範囲に正規化（負の値は0、正の値は高く）
        scores["Abstractness"] = max(0, min(1.0, abstractness_raw + 0.3))
    
    # ===== 10. 多角性 =====
    multi = features_sum["marker_perspective"] * 1.5
    multi += features_sum["marker_multiple_options"] * 1.3
    multi += features_sum["marker_anticipate"]
    scores["Multi_Perspective"] = multi / total_count
    
    # ===== 11. 内省性 =====
    reflective = features_sum["marker_self_reference"] * 0.8
    reflective += features_sum["marker_mental_process"] * 1.5
    reflective += features_sum["marker_self_improvement"] * 1.8
    scores["Self_Reflection"] = reflective / total_count
    
    # ===== 12. 未来志向性 =====
    future = features_sum["marker_future_time"] * 1.2
    future += features_sum["marker_planning"] * 1.5
    future += features_sum["marker_possibility"]
    scores["Future_Oriented"] = future / total_count
    
    # ===== 13. リスク感知性 =====
    risk = features_sum["marker_risk"] * 1.5
    risk += features_sum["marker_conditional"]
    risk += features_sum["marker_failure"] * 1.2
    # 対策語があると健全（少し加点）
    risk += features_sum["marker_countermeasure"] * 0.5
    scores["Risk_Awareness"] = risk / total_count
    
    return scores


def calculate_confidence(msg_count: int, char_count: int) -> float:
    """
    信頼度を計算（0.0〜1.0）
    メッセージ数と文字数の両方を考慮
    """
    # メッセージ数ベース（最低20件、理想200件以上）
    msg_factor = min(1.0, msg_count / 200)
    
    # 文字数ベース（最低500文字、理想5000文字以上）
    char_factor = min(1.0, char_count / 5000)
    
    # 両方の平均
    confidence = (msg_factor + char_factor) / 2
    
    return round(confidence, 2)


# ─────────────────────────────────────
# 後方互換性のためのダミー関数
# ─────────────────────────────────────

def classify_to_json(text: str) -> dict:
    """
    後方互換性のため（使用されていない場合は削除可能）
    """
    return {
        "comm_style": "Lead_Directiveness",
        "think_style": "Structural_Thinking"
    }