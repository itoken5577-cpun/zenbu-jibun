"""
aggregate.py - 集計ロジック（相手別スタイル分布・全体平均・差分）
"""
import json
from typing import Dict, List, Any, Tuple

import pandas as pd

from classify_rules import COMM_STYLE_LABELS, THINK_STYLE_LABELS


def build_distribution(messages: List[Dict[str, Any]]) -> Dict[str, Dict]:
    """
    戻り値:
    {
      "global": {"style_dist": {...}, "think_dist": {...}, "count": N},
      "<counterparty>": {...},
      ...
    }
    """
    if not messages:
        return {"global": _empty_dist(0)}

    df = pd.DataFrame(messages)
    df = df.dropna(subset=["style_primary", "think_primary"])
    if df.empty:
        return {"global": _empty_dist(0)}

    result: Dict[str, Dict] = {}
    result["global"] = _calc_dist(df)
    for cp in df["counterparty"].unique():
        result[cp] = _calc_dist(df[df["counterparty"] == cp])

    return result


def _empty_dist(count: int) -> Dict:
    return {
        "style_dist": {k: 0.0 for k in COMM_STYLE_LABELS},
        "think_dist": {k: 0.0 for k in THINK_STYLE_LABELS},
        "count": count,
    }


def _calc_dist(df: pd.DataFrame) -> Dict:
    count = len(df)
    if count == 0:
        return _empty_dist(0)

    style_counts = df["style_primary"].value_counts()
    think_counts = df["think_primary"].value_counts()

    style_dist = {
        label: float(round(style_counts.get(label, 0) / count, 4))
        for label in COMM_STYLE_LABELS
    }
    think_dist = {
        label: float(round(think_counts.get(label, 0) / count, 4))
        for label in THINK_STYLE_LABELS
    }

    return {
        "style_dist": style_dist,
        "think_dist": think_dist,
        "count": int(count),
    }


def calc_diff_from_global(dist_result: Dict[str, Dict]) -> Dict[str, Dict]:
    """各 counterparty の全体平均との差分を計算"""
    global_style = dist_result.get("global", {}).get("style_dist", {})
    global_think = dist_result.get("global", {}).get("think_dist", {})

    diffs: Dict[str, Dict] = {}
    for cp, dist in dist_result.items():
        if cp == "global":
            continue
        style_diff = {
            label: round(dist["style_dist"].get(label, 0) - global_style.get(label, 0), 4)
            for label in COMM_STYLE_LABELS
        }
        think_diff = {
            label: round(dist["think_dist"].get(label, 0) - global_think.get(label, 0), 4)
            for label in THINK_STYLE_LABELS
        }
        diffs[cp] = {"style_diff": style_diff, "think_diff": think_diff}
    return diffs


def top3_diff(diffs: Dict[str, Dict], counterparty: str) -> List[Dict]:
    """指定 counterparty の差分 Top3（絶対値が大きいもの）"""
    if counterparty not in diffs:
        return []
    cp_diffs = diffs[counterparty]
    items = []
    for label, val in cp_diffs["style_diff"].items():
        items.append({"label": label, "kind": "コミュニケーション", "diff": val})
    for label, val in cp_diffs["think_diff"].items():
        items.append({"label": label, "kind": "思考", "diff": val})
    items.sort(key=lambda x: abs(x["diff"]), reverse=True)
    return items[:3]


def build_summary_json(
    dist_result: Dict[str, Dict],
    diffs: Dict[str, Dict],
    my_name: str,
) -> Dict:
    """外部LLMに渡す集計JSON（生ログ不含）"""
    summary = {
        "meta": {
            "my_name_hash": f"user_{hash(my_name) % 99999:05d}",
            "note": "This JSON contains only aggregated statistics. No raw message content is included.",
        },
        "global": dist_result.get("global", {}),
        "by_counterparty": {},
    }
    for cp, dist in dist_result.items():
        if cp == "global":
            continue
        cp_hash = f"room_{hash(cp) % 99999:05d}"
        summary["by_counterparty"][cp_hash] = {
            "display_name": cp,
            "count": dist["count"],
            "style_dist": dist["style_dist"],
            "think_dist": dist["think_dist"],
            "diff_from_global": diffs.get(cp, {}),
            "top3_diff": top3_diff(diffs, cp),
        }
    return summary


def dist_to_dataframe(
    dist_result: Dict[str, Dict],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """dist_result を pandas DataFrame に変換（可視化用）"""
    style_rows = []
    think_rows = []
    for cp, dist in dist_result.items():
        row_s = {"counterparty": cp, "count": dist["count"]}
        row_s.update(dist["style_dist"])
        style_rows.append(row_s)

        row_t = {"counterparty": cp, "count": dist["count"]}
        row_t.update(dist["think_dist"])
        think_rows.append(row_t)

    df_style = pd.DataFrame(style_rows).set_index("counterparty")
    df_think = pd.DataFrame(think_rows).set_index("counterparty")
    return df_style, df_think
```

---

## `prompts/insight_prompt.txt`
```
# 全部自分 - インサイト生成プロンプト
# 外部LLM（ChatGPT, Claude等）に貼り付けて使用してください。
# ※ このプロンプトに続けて「集計JSON」を貼り付けてください。

---

あなたはコミュニケーションスタイルの分析アシスタントです。
以下の集計JSONは、あるユーザーのLINEトーク履歴から生成されたものです。
（生のメッセージ内容は含まれていません。統計データのみです。）

【重要な制約】
- 断定的な診断・評価・ラベリングは行わないでください。
- 「あなたは〇〇な人です」「このスタイルは問題です」のような表現は使わないでください。
- あくまで「傾向の観察」と「問い」を提示するにとどめてください。

【出力形式】
以下の順序で、日本語で回答してください。

## 1. 事実の観察
数字や比率をそのまま読み取り、客観的に描写してください。
例：「Askの比率が全体平均より15%高い傾向が見られます」

## 2. 可能な解釈（複数）
その数値の背景にある可能性のある解釈を、2〜3個提示してください。
断定せず、「〜かもしれません」「〜という可能性が考えられます」の形で。

## 3. 気づきのための問い
ユーザー自身が内省できるような、オープンな問いを2〜3個立ててください。
例：「この相手との会話で、自分はどんな役割を担っていると感じていますか？」

---

【集計JSON（ここに貼り付けてください）】
```

---

## `sample_data/sample_line.txt`
```
[LINE] 仕事チームのトーク履歴
保存日時：2024/01/15 20:00

2024/01/10(水)

10:30	健悟	おはよう！今日のミーティング何時から？
10:31	田中	10時からだよ。もう準備できてる？
10:32	健悟	了解。ちょっと資料確認してからいく
10:35	田中	ちなみに今日のアジェンダってどんな感じ？
10:36	健悟	まず先週の進捗確認、次に来週のスケジュール調整、最後にQ1の目標設定の話だよ
10:37	田中	なるほど。目標設定の資料とかある？
10:38	健悟	あるよ。Googleドライブにあげてあるから確認しておいてほしい
10:39	健悟	なぜなら、今日の議論で数字を使うから事前に見ておいた方がいいと思う
10:40	田中	わかった。ありがとう
10:45	健悟	もし不明点あったら事前にメッセージして。ミーティング中にスムーズに進めたいから

2024/01/11(木)

14:00	健悟	昨日のミーティング、お疲れ様でした
14:01	鈴木	お疲れ様でした！なかなか盛り上がりましたね
14:02	健悟	そうだね笑　でも良い議論ができたと思う。鈴木さんの意見、すごく参考になった
14:03	鈴木	ありがとうございます。健悟さんのファシリテーションがうまかったですよ
14:05	健悟	いやいや、みんなが積極的に話してくれたおかげだよ。今後も同じペースで進めていきたいね
14:06	健悟	ところで、次回は来週水曜でどう？
14:07	鈴木	水曜ですか。大丈夫そうです
14:08	健悟	じゃあ確定で。議題は今回の続きで、具体的なタスク割り振りにしましょう
14:09	鈴木	了解です。楽しみにしてます

2024/01/12(金)

09:00	健悟	田中さん、昨日のレポート確認したよ
09:01	田中	どうでしたか？
09:02	健悟	全体的にいい内容だった！ただリスクの部分、もう少し具体的に書いてほしいな
09:03	健悟	例えば、遅延した場合の代替案とか、万が一の対策とか
09:04	田中	なるほど、確かに曖昧でしたね。修正します
09:05	健悟	お願い。締め切りは明日の17時までにできる？
09:06	田中	はい、対応します
09:07	健悟	ありがとう。不明点あれば聞いてね

2024/01/13(土)

16:30	健悟	週末に連絡してごめん
16:31	鈴木	大丈夫ですよ〜、何ですか？
16:32	健悟	来週の件なんだけど、新しいアイデアが浮かんだから試してみたくて
16:33	健悟	競合調査の方法を変えてみようかと思っているんだけど、どう思う？
16:34	鈴木	面白そうですね！どんな方法ですか？
16:35	健悟	ユーザーインタビューを先にやって、そこから逆算して競合を見るやり方
16:36	健悟	通常は競合から入るけど、ユーザー視点を先に持てる可能性があると思って
16:37	鈴木	確かに新しい視点ですね。可能性ありそう
16:38	健悟	まず小さく試してみない？来週のミーティングで提案しようと思ってるんだけど
16:39	鈴木	いいと思います！ぜひやってみましょう
```

---

## `requirements.txt`
```
streamlit>=1.32.0
pandas>=2.0.0
scikit-learn>=1.4.0
plotly>=5.18.0
altair>=5.2.0
