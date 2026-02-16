"""
classify_rules.py - ルールベースのスタイル/思考スタイル分類
スコア方式: キーワード・記号・構造で各ラベルにスコアを付与
"""
import re
import json
from typing import Dict, Tuple, List

# ======================================================
# コミュニケーションスタイル（8ラベル）
# ======================================================
COMM_STYLE_RULES: Dict[str, dict] = {
    "Ask": {
        "patterns": [
            r"[？?]",
            r"ですか[？?]?",
            r"でしょうか",
            r"どう(思う|ですか|でしょう)",
            r"(教えて|聞かせて|確認)",
            r"ですよね[？?]?",
            r"(いつ|どこ|誰|何|なぜ|どれ|どうして|どんな)",
            r"(知ってる|わかる|できる)\??",
            r"かな[？?]?$",
        ],
        "bonus_patterns": [r"[？?].*[？?]"],
        "weights": [2, 2, 2, 2, 2, 1, 2, 1, 1],
        "bonus_weights": [1],
    },
    "Propose": {
        "patterns": [
            r"(どうかな|どうでしょう|いかがでしょう)",
            r"(提案|案|アイデア|考え)",
            r"(してみ(ては|たら)|やってみ(ては|たら))",
            r"(試して|試みて|試してみ)",
            r"(〜はどう|〜はいかが|〜にしない[?？]?)",
            r"(こういう|こんな感じ)(で|に)(は|も)?",
            r"(方がいい|ほうがいい|のでは|と思う)",
            r"(案として|方法として|選択肢)",
        ],
        "weights": [2, 2, 2, 1, 2, 1, 1, 2],
    },
    "Structure": {
        "patterns": [
            r"^\s*[①②③④⑤⑥⑦⑧⑨⑩]",
            r"^\s*[1-9][.．、。]",
            r"^\s*[-・•▶▷→]\s",
            r"(まず|次に|最後に|第[一二三四五]に)",
            r"(整理すると|まとめると|要点は|ポイントは)",
            r"(以下の通り|以下をご確認|下記の)",
            r"(項目|リスト|一覧|ステップ)",
            r"\n.*\n.*\n",
        ],
        "weights": [3, 3, 2, 2, 3, 2, 2, 2],
    },
    "Empathize": {
        "patterns": [
            r"(つらい|辛い|大変|しんどい|疲れ)",
            r"(嬉しい|うれしい|よかった|ありがとう|感謝)",
            r"(心配|不安|怖い|こわい)",
            r"(わかる|わかるよ|わかります|共感)",
            r"(ほんとに|本当に|本当うれし|本当よかった)",
            r"(お疲れ|お疲れ様|ねぎら)",
            r"(大丈夫[？?]?|気をつけて|無理しないで)",
            r"(応援|サポート|力になる|一緒に)",
            r"[！!]{2,}",
        ],
        "weights": [2, 2, 2, 2, 2, 2, 2, 2, 1],
    },
    "Explain": {
        "patterns": [
            r"(なぜなら|理由は|背景は|というのも)",
            r"(つまり|すなわち|言い換えると|要するに)",
            r"(例えば|たとえば|具体的には)",
            r"(仕組み|メカニズム|原因|結果)",
            r"(説明|解説|詳細|補足)",
            r"(〜とは|というのは|意味は)",
            r"(なお|ちなみに|補足すると|付け加えると)",
            r"(〜のため|〜によって|〜の結果)",
        ],
        "weights": [2, 2, 2, 2, 2, 2, 1, 1],
    },
    "Lead": {
        "patterns": [
            r"(してください|お願いします|お願いいたします)",
            r"(やっておいて|対応して|確認して)",
            r"(決定|決めた|確定|決まった|方針)",
            r"(〜すべき|〜必要|〜しないと|〜してほしい)",
            r"(担当|役割|責任|タスク)",
            r"(指示|連絡|報告|共有)して",
            r"(期限|締め切り|デッドライン|いつまでに)",
            r"(進めて|進捗|対応|実行)",
        ],
        "weights": [2, 2, 2, 2, 2, 2, 2, 2],
    },
    "Align": {
        "patterns": [
            r"^(了解|りょ|OK|ok|承知|承りました|わかりました|わかった)",
            r"(そうですね|そうだね|そうかも|たしかに|確かに)",
            r"(賛成|同意|同感|おっしゃる通り)",
            r"(合わせます|調整します|検討します|考えます)",
            r"(〜ですね[！!。]|〜ですよね)",
            r"(いいね[！!]?|いいと思|良いと思)",
            r"(よろしく(お願いします)?|よろです)",
        ],
        "weights": [3, 2, 2, 2, 1, 2, 2],
    },
    "SmallTalk": {
        "patterns": [
            r"^(おはよ|おはよう|こんにちは|こんばんは|お疲れ|ただいま|おやすみ)",
            r"(笑|w+|草|ｗ+)",
            r"(なんか|なんだか|なんとなく)",
            r"(今日|昨日|明日)(は|も|の)?(どう|何|暇|時間)",
            r"(〜じゃん|〜だよね|〜だよ|〜だね)$",
            r"(ご飯|飲み|遊び|映画|買い物)(いこ|しよ|どう)",
            r"(ありがとう|さんきゅ|ありがと)[！!。]?$",
            r"(久しぶり|お久しぶり)",
        ],
        "weights": [3, 2, 1, 2, 1, 2, 2, 2],
    },
}

# ======================================================
# 思考スタイル（6ラベル）
# ======================================================
THINK_STYLE_RULES: Dict[str, dict] = {
    "Logic": {
        "patterns": [
            r"(なぜなら|理由|原因|根拠|証拠|データ)",
            r"(したがって|よって|ゆえに|そのため|だから)",
            r"(AならばB|もし〜なら|前提|仮定)",
            r"(比較|対比|一方|他方|逆に|反対に)",
            r"(論理|筋道|整合|矛盾|一貫)",
            r"(分析|検証|考察|評価|判断)",
            r"(数字|数値|割合|パーセント|統計)",
            r"\d+(\.\d+)?[%％]",
        ],
        "weights": [2, 2, 2, 2, 2, 2, 2, 2],
    },
    "Other": {
        "patterns": [
            r"(相手|あなた|あなたの|君|あなたが)(は|に|の|が|を)",
            r"(みんな|周り|チーム|メンバー|仲間)",
            r"(立場|視点|目線|観点|感じ方)",
            r"(相手の気持ち|相手目線|他者|ユーザー)",
            r"(配慮|気遣い|思いやり|気にする)",
            r"(伝わる|伝えたい|わかってもらえる|理解してもらえる)",
        ],
        "weights": [2, 2, 2, 3, 2, 2],
    },
    "Goal": {
        "patterns": [
            r"(目標|目的|ゴール|KPI|指標|成果)",
            r"(達成|クリア|完了|完成|実現)",
            r"(〜するために|〜を目指して|〜のために)",
            r"(戦略|計画|ロードマップ|スケジュール)",
            r"(優先|最優先|まず|まずは|一番大事)",
            r"(成功|成果|結果|アウトプット|output)",
            r"(期待|期待値|見込み|予定)",
        ],
        "weights": [3, 2, 2, 2, 2, 2, 1],
    },
    "Risk": {
        "patterns": [
            r"(リスク|危険|危機|問題|課題|懸念|懸念点)",
            r"(失敗|ミス|エラー|バグ|不具合)",
            r"(注意|気をつけ|確認して|チェック)(が必要|しないと|しよう)?",
            r"(〜しないと|〜してしまう|〜になりかねない)",
            r"(対策|回避|防止|予防|最悪)",
            r"(遅延|遅れ|超過|オーバー|間に合わない)",
            r"(想定外|予期しない|万が一|もしも)",
        ],
        "weights": [3, 2, 2, 2, 2, 2, 2],
    },
    "Explore": {
        "patterns": [
            r"(面白い|興味深い|気になる|試してみたい)",
            r"(新しい|新機能|最新|トレンド|話題)",
            r"(学んだ|調べた|発見|気づき|へえ|ほう)",
            r"(可能性|潜在|応用|使えそう|活かせる)",
            r"(どうなんだろ|どんな感じ|試してみよう|やってみたい)",
            r"(実験|プロトタイプ|PoC|アイデア|ひらめき)",
            r"(勉強|インプット|読んだ|調査|リサーチ)",
        ],
        "weights": [2, 2, 2, 2, 2, 2, 2],
    },
    "Stability": {
        "patterns": [
            r"(いつも通り|通常通り|今まで通り|例年通り)",
            r"(安定|継続|維持|キープ|持続)",
            r"(ルール|規則|ポリシー|基準|ガイドライン)",
            r"(慣れ|習慣|定常|定例|定期)",
            r"(変えない|変えず|今のまま|現状維持)",
            r"(確実に|着実に|堅実に|丁寧に)",
            r"(実績|前例|経験|過去に|これまでも)",
        ],
        "weights": [2, 2, 2, 2, 3, 2, 2],
    },
}


def _compile_rules(rules_dict: Dict) -> Dict:
    compiled = {}
    for label, cfg in rules_dict.items():
        compiled[label] = {
            "patterns": [
                re.compile(p, re.IGNORECASE | re.MULTILINE)
                for p in cfg.get("patterns", [])
            ],
            "weights": cfg.get("weights", [1] * len(cfg.get("patterns", []))),
            "bonus_patterns": [
                re.compile(p, re.IGNORECASE | re.MULTILINE)
                for p in cfg.get("bonus_patterns", [])
            ],
            "bonus_weights": cfg.get("bonus_weights", []),
        }
    return compiled


_COMM_COMPILED = _compile_rules(COMM_STYLE_RULES)
_THINK_COMPILED = _compile_rules(THINK_STYLE_RULES)


def _score_text(text: str, compiled_rules: Dict) -> Dict[str, float]:
    scores = {}
    for label, cfg in compiled_rules.items():
        score = 0.0
        for pat, w in zip(cfg["patterns"], cfg["weights"]):
            if pat.search(text):
                score += w
        for pat, w in zip(cfg["bonus_patterns"], cfg["bonus_weights"]):
            if pat.search(text):
                score += w
        scores[label] = score
    return scores


def classify_message(text: str) -> Tuple:
    """
    戻り値: (style_primary, think_primary, style_scores, think_scores)
    """
    style_scores = _score_text(text, _COMM_COMPILED)
    think_scores = _score_text(text, _THINK_COMPILED)

    style_max = max(style_scores.values()) if style_scores else 0
    think_max = max(think_scores.values()) if think_scores else 0

    style_primary = "SmallTalk" if style_max == 0 else max(style_scores, key=style_scores.get)
    think_primary = "Other" if think_max == 0 else max(think_scores, key=think_scores.get)

    return style_primary, think_primary, style_scores, think_scores


def classify_to_json(text: str) -> Dict:
    sp, tp, ss, ts = classify_message(text)
    return {
        "style_primary": sp,
        "think_primary": tp,
        "style_score_json": json.dumps(ss, ensure_ascii=False),
        "think_score_json": json.dumps(ts, ensure_ascii=False),
    }


# ラベル一覧（順序固定）
COMM_STYLE_LABELS = ["Ask", "Propose", "Structure", "Empathize", "Explain", "Lead", "Align", "SmallTalk"]
THINK_STYLE_LABELS = ["Logic", "Other", "Goal", "Risk", "Explore", "Stability"]
