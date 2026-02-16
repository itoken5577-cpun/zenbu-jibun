"""
aggregate.py - 集計ロジック（相手別スタイル分布・全体平均・差分）
"""
from __future__ import annotations

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
