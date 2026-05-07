"""
管道健康度评分模块
基于变形率、风险等级、帧占比计算 0-100 综合健康评分
"""


def calc_health_score(results, risk_threshold_high=None, risk_threshold_mid=None):
    """
    计算管道健康度评分。
    """
    if not results:
        return _empty_score()

    from database import db_ops
    # 获取动态阈值和权重
    try:
        th_high = float(risk_threshold_high if risk_threshold_high is not None else db_ops.get_config("risk_threshold_high"))
        th_mid  = float(risk_threshold_mid if risk_threshold_mid is not None else db_ops.get_config("risk_threshold_mid"))
        
        weight_high = float(db_ops.get_config("health_penalty_high"))
        weight_mid  = float(db_ops.get_config("health_penalty_mid"))
        limit_high  = float(db_ops.get_config("health_limit_high", 60.0))
        limit_mid   = float(db_ops.get_config("health_limit_mid", 25.0))
    except (TypeError, ValueError):
        # 如果数据库没有这些配置，则使用内置硬编码（仅作为最后防线）
        th_high, th_mid = 0.08, 0.03
        weight_high, weight_mid = 1.5, 0.5
        limit_high, limit_mid = 60.0, 25.0

    total = len(results)
    high_cnt = mid_cnt = low_cnt = 0
    deform_vals = []

    for item in results:
        risk = (item.get("RiskLevel") or "").strip()
        if risk == "高":
            high_cnt += 1
        elif risk == "中":
            mid_cnt += 1
        else:
            low_cnt += 1

        d = item.get("Deformation")
        try:
            d = float(d)
            if 0.0 <= d <= 1.0:
                deform_vals.append(d)
        except (TypeError, ValueError):
            pass

    high_ratio = high_cnt / total
    mid_ratio  = mid_cnt  / total
    low_ratio  = low_cnt  / total
    avg_deform = sum(deform_vals) / len(deform_vals) if deform_vals else 0.0
    max_deform = max(deform_vals) if deform_vals else 0.0

    # ── 扣分规则 ──────────────────────────────────────────────
    # 高风险帧占比：每 1% 扣 weight_high 分，上限扣 limit_high 分
    penalty_high = min(high_ratio * 100 * weight_high, limit_high)
    # 中风险帧占比：每 1% 扣 weight_mid 分，上限扣 limit_mid 分
    penalty_mid  = min(mid_ratio  * 100 * weight_mid, limit_mid)
    # 平均变形率超过中风险阈值的部分额外扣分
    deform_excess = max(avg_deform - th_mid, 0.0)
    penalty_deform = min(deform_excess / (th_high - th_mid + 1e-6) * 15.0, 15.0)

    raw_score = 100.0 - penalty_high - penalty_mid - penalty_deform
    score = max(0, min(100, round(raw_score)))

    if score >= 75:
        level, color = "优良", "green"
    elif score >= 45:
        level, color = "警告", "orange"
    else:
        level, color = "危险", "red"

    detail = (
        f"高风险帧 {high_ratio*100:.1f}%（扣 {penalty_high:.1f} 分）｜"
        f"中风险帧 {mid_ratio*100:.1f}%（扣 {penalty_mid:.1f} 分）｜"
        f"变形率惩罚（扣 {penalty_deform:.1f} 分）"
    )

    return {
        "score":      score,
        "level":      level,
        "color":      color,
        "high_ratio": high_ratio,
        "mid_ratio":  mid_ratio,
        "low_ratio":  low_ratio,
        "avg_deform": avg_deform,
        "max_deform": max_deform,
        "detail":     detail,
    }


def _empty_score():
    return {
        "score": 100, "level": "优良", "color": "green",
        "high_ratio": 0.0, "mid_ratio": 0.0, "low_ratio": 0.0,
        "avg_deform": 0.0, "max_deform": 0.0, "detail": "无检测数据",
    }
