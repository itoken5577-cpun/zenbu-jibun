"""
aggregate.py - 集計・分析（新13軸対応）
"""
from collections import defaultdict
from typing import List, Dict, Any
import json


def build_distribution(messages: List[Dict]) -> Dict:
    """
    メッセージから相手別・グローバルの分布を計算
    新13軸対応版
    """
    from classify_rules import (
        calculate_axis_scores,
        calculate_confidence,
        COMM_STYLE_LABELS,
        THINK_STYLE_LABELS,
    )
    
    result = {}
    
    # 自分のメッセージのみ
    my_messages = [m for m in messages if m.get("is_me")]
    
    if not my_messages:
        return {"global": {"count": 0}}
    
    # ===== グローバル集計 =====
    global_scores = calculate_axis_scores(my_messages)
    global_confidence = calculate_confidence(
        len(my_messages),
        sum(len(m.get("text", "")) for m in my_messages)
    )
    
    # コミュニケーションと思考に分ける
    comm_dist = {k: global_scores[k] for k in COMM_STYLE_LABELS}
    think_dist = {k: global_scores[k] for k in THINK_STYLE_LABELS}
    
    result["global"] = {
        "count": len(my_messages),
        "style_dist": comm_dist,
        "think_dist": think_dist,
        "confidence": global_confidence,
    }
    
    # ===== 相手別集計 =====
    by_counterparty = defaultdict(list)
    for m in my_messages:
        cp = m.get("counterparty", "unknown")
        by_counterparty[cp].append(m)
    
    for cp, cp_messages in by_counterparty.items():
        cp_scores = calculate_axis_scores(cp_messages)
        cp_confidence = calculate_confidence(
            len(cp_messages),
            sum(len(m.get("text", "")) for m in cp_messages)
        )
        
        comm_dist_cp = {k: cp_scores[k] for k in COMM_STYLE_LABELS}
        think_dist_cp = {k: cp_scores[k] for k in THINK_STYLE_LABELS}
        
        result[cp] = {
            "count": len(cp_messages),
            "style_dist": comm_dist_cp,
            "think_dist": think_dist_cp,
            "confidence": cp_confidence,
        }
    
    return result


def calc_diff_from_global(dist_result: Dict) -> Dict:
    """
    各相手とグローバル平均との差分を計算
    """
    from classify_rules import COMM_STYLE_LABELS, THINK_STYLE_LABELS
    
    global_data = dist_result.get("global", {})
    g_style = global_data.get("style_dist", {})
    g_think = global_data.get("think_dist", {})
    
    diffs = {}
    
    for cp, cp_data in dist_result.items():
        if cp == "global":
            continue
        
        cp_style = cp_data.get("style_dist", {})
        cp_think = cp_data.get("think_dist", {})
        
        # 差分計算
        style_diffs = {k: cp_style.get(k, 0) - g_style.get(k, 0) for k in COMM_STYLE_LABELS}
        think_diffs = {k: cp_think.get(k, 0) - g_think.get(k, 0) for k in THINK_STYLE_LABELS}
        
        diffs[cp] = {
            "style_diffs": style_diffs,
            "think_diffs": think_diffs,
            "confidence": cp_data.get("confidence", 0),
        }
    
    return diffs


def top3_diff(diffs_all: Dict, counterparty: str, top_n: int = 3) -> List[Dict]:
    """
    指定した相手のTop3差分を取得
    """
    from classify_rules import COMM_STYLE_DISPLAY, THINK_STYLE_DISPLAY
    
    cp_diff = diffs_all.get(counterparty)
    if not cp_diff:
        return []
    
    style_diffs = cp_diff.get("style_diffs", {})
    think_diffs = cp_diff.get("think_diffs", {})
    
    # 全ての差分を集める
    all_diffs = []
    
    for label, diff in style_diffs.items():
        all_diffs.append({
            "kind": "comm",  # コミュニケーション
            "label": label,
            "display": COMM_STYLE_DISPLAY.get(label, label),
            "diff": diff,
        })
    
    for label, diff in think_diffs.items():
        all_diffs.append({
            "kind": "think",  # 思考
            "label": label,
            "display": THINK_STYLE_DISPLAY.get(label, label),
            "diff": diff,
        })
    
    # 絶対値の大きい順にソート
    all_diffs.sort(key=lambda x: abs(x["diff"]), reverse=True)
    
    return all_diffs[:top_n]


def dist_to_dataframe(dist_result: Dict):
    """
    分布をDataFrameに変換（グラフ表示用）
    """
    import pandas as pd
    from classify_rules import COMM_STYLE_LABELS, THINK_STYLE_LABELS
    
    rows_style = []
    rows_think = []
    
    for cp, data in dist_result.items():
        style_dist = data.get("style_dist", {})
        think_dist = data.get("think_dist", {})
        
        row_style = {"counterparty": cp}
        row_style.update(style_dist)
        rows_style.append(row_style)
        
        row_think = {"counterparty": cp}
        row_think.update(think_dist)
        rows_think.append(row_think)
    
    df_style = pd.DataFrame(rows_style).set_index("counterparty")
    df_think = pd.DataFrame(rows_think).set_index("counterparty")
    
    # カラムの順序を保証
    df_style = df_style[[col for col in COMM_STYLE_LABELS if col in df_style.columns]]
    df_think = df_think[[col for col in THINK_STYLE_LABELS if col in df_think.columns]]
    
    return df_style, df_think


def build_summary_json(dist_result: Dict, diffs_all: Dict, my_name: str = "ユーザー") -> Dict:
    """
    LLMエクスポート用のJSONを生成
    """
    from classify_rules import COMM_STYLE_DISPLAY, THINK_STYLE_DISPLAY
    
    summary = {
        "user_name": my_name,
        "global": {},
        "counterparties": [],
    }
    
    # グローバル
    g = dist_result.get("global", {})
    if g:
        summary["global"] = {
            "message_count": g.get("count", 0),
            "confidence": g.get("confidence", 0),
            "communication_styles": {
                COMM_STYLE_DISPLAY[k]: round(v, 3)
                for k, v in g.get("style_dist", {}).items()
            },
            "thinking_styles": {
                THINK_STYLE_DISPLAY[k]: round(v, 3)
                for k, v in g.get("think_dist", {}).items()
            },
        }
    
    # 相手別
    for cp, data in dist_result.items():
        if cp == "global":
            continue
        
        cp_summary = {
            "name": cp,
            "message_count": data.get("count", 0),
            "confidence": data.get("confidence", 0),
            "communication_styles": {
                COMM_STYLE_DISPLAY[k]: round(v, 3)
                for k, v in data.get("style_dist", {}).items()
            },
            "thinking_styles": {
                THINK_STYLE_DISPLAY[k]: round(v, 3)
                for k, v in data.get("think_dist", {}).items()
            },
            "top3_differences": top3_diff(diffs_all, cp, 3),
        }
        
        summary["counterparties"].append(cp_summary)
    
    return summary