from Library import NP, CV2, App, DataFrame


def _to_binary_mask(mask):
    if mask is None:
        return None
    if mask.ndim > 2:
        mask = mask[:, :, 0]
    return (mask > 127).astype(NP.uint8)


def _mask_iou(mask_a, mask_b):
    if mask_a is None or mask_b is None:
        return 0.0
    inter = NP.logical_and(mask_a == 1, mask_b == 1).sum()
    union = NP.logical_or(mask_a == 1, mask_b == 1).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def build_single_explainability(model_name, raw_mask, avg_confidence=None):
    binary_mask = _to_binary_mask(raw_mask)
    area_ratio = float(binary_mask.mean()) if binary_mask is not None else 0.0
    return {
        "type": "single",
        "model_name": model_name,
        "avg_confidence": avg_confidence,
        "mask_area_ratio": area_ratio,
    }


def build_ensemble_explainability(image_bgr, model_masks, fused_mask):
    valid_masks = {name: _to_binary_mask(mask) for name, mask in model_masks.items() if mask is not None}
    if not valid_masks:
        return {
            "type": "ensemble",
            "agreement_ratio": 0.0,
            "pairwise_iou": [],
            "vote_distribution": {},
            "uncertainty_overlay": None,
            "model_area": [],
        }

    names = list(valid_masks.keys())
    stack = NP.stack([valid_masks[name] for name in names], axis=0)
    model_count = stack.shape[0]
    votes = stack.sum(axis=0)

    agreement_map = NP.logical_or(votes == 0, votes == model_count)
    agreement_ratio = float(agreement_map.mean())

    normalized_vote = votes / max(model_count, 1)
    uncertainty = 1.0 - NP.abs(normalized_vote - 0.5) * 2.0
    uncertainty[agreement_map] = 0.0

    disagreement_mask = ((votes > 0) & (votes < model_count)).astype(NP.uint8)
    disagreement_ratio = float(disagreement_mask.mean())

    amplified = NP.power(uncertainty, 0.35)
    amplified = NP.clip(amplified * 1.35, 0.0, 1.0)
    amplified[disagreement_mask == 1] = NP.maximum(amplified[disagreement_mask == 1], 0.9)

    amplified_img = (amplified * 255).astype(NP.uint8)
    amplified_img = CV2.dilate(amplified_img, NP.ones((3, 3), dtype=NP.uint8), iterations=1)
    heatmap = CV2.applyColorMap(amplified_img, CV2.COLORMAP_TURBO)

    overlay = None
    if image_bgr is not None:
        if image_bgr.ndim == 2:
            image_bgr = CV2.cvtColor(image_bgr, CV2.COLOR_GRAY2BGR)
        if image_bgr.shape[:2] != heatmap.shape[:2]:
            heatmap = CV2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=CV2.INTER_LINEAR)
        overlay_bgr = CV2.addWeighted(image_bgr, 0.45, heatmap, 0.95, 0)

        contours, _ = CV2.findContours((disagreement_mask * 255).astype(NP.uint8), CV2.RETR_EXTERNAL, CV2.CHAIN_APPROX_SIMPLE)
        if contours:
            CV2.drawContours(overlay_bgr, contours, -1, (255, 255, 255), 2)

        overlay = CV2.cvtColor(overlay_bgr, CV2.COLOR_BGR2RGB)

    pairwise_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            left_name = names[i]
            right_name = names[j]
            pairwise_rows.append(
                {
                    "ModelA": left_name,
                    "ModelB": right_name,
                    "MaskIoU": round(_mask_iou(valid_masks[left_name], valid_masks[right_name]), 4),
                }
            )

    vote_distribution = {
        "all_background": int((votes == 0).sum()),
        "split_pixels": int(((votes > 0) & (votes < model_count)).sum()),
        "all_foreground": int((votes == model_count).sum()),
    }

    model_area = []
    for name in names:
        model_area.append({"Model": name, "MaskAreaRatio": round(float(valid_masks[name].mean()), 4)})

    fused_area_ratio = float(_to_binary_mask(fused_mask).mean()) if fused_mask is not None else 0.0

    return {
        "type": "ensemble",
        "agreement_ratio": agreement_ratio,
        "disagreement_ratio": disagreement_ratio,
        "pairwise_iou": pairwise_rows,
        "vote_distribution": vote_distribution,
        "uncertainty_overlay": overlay,
        "model_area": model_area,
        "fused_area_ratio": fused_area_ratio,
    }


def render_explainability_panel(explain_data, use_expander=True):
    if not explain_data:
        return

    def _build_auto_summary(data):
        if data.get("type") == "ensemble":
            agreement = float(data.get("agreement_ratio", 0.0))
            split_pixels = int(data.get("vote_distribution", {}).get("split_pixels", 0))
            if agreement >= 0.95:
                level = "高一致"
                advice = "三模型结论高度一致，可直接作为自动判定结果。"
            elif agreement >= 0.85:
                level = "中一致"
                advice = "整体一致性较好，建议重点关注边缘区域。"
            else:
                level = "低一致"
                advice = "模型分歧较明显，建议人工复核或重新采样。"
            return f"解释结论：{level}（一致率 {agreement * 100:.2f}%），分歧像素 {split_pixels}。{advice}"

        area_ratio = float(data.get("mask_area_ratio", 0.0))
        confidence = data.get("avg_confidence")
        if confidence is None:
            confidence_text = "置信度不可用"
            confidence_low = False
        else:
            confidence_text = f"平均置信度 {confidence:.4f}"
            confidence_low = confidence < 0.5

        if area_ratio < 0.01:
            area_text = "目标区域很小"
        elif area_ratio < 0.15:
            area_text = "目标区域适中"
        else:
            area_text = "目标区域较大"

        if confidence_low:
            advice = "建议人工复核该结果。"
        else:
            advice = "结果稳定，可作为当前判定依据。"

        return f"解释结论：{area_text}（掩码占比 {area_ratio * 100:.2f}%），{confidence_text}。{advice}"

    def _render_body():
        if explain_data.get("type") == "ensemble":
            col1, col2, col3, col4 = App.columns(4)
            with col1:
                App.metric("模型一致率", f"{explain_data.get('agreement_ratio', 0.0) * 100:.2f}%")
            with col2:
                App.metric("模型分歧率", f"{explain_data.get('disagreement_ratio', 0.0) * 100:.2f}%")
            with col3:
                App.metric("融合掩码占比", f"{explain_data.get('fused_area_ratio', 0.0) * 100:.2f}%")
            with col4:
                split_pixels = explain_data.get("vote_distribution", {}).get("split_pixels", 0)
                App.metric("分歧像素数", split_pixels)

            overlay = explain_data.get("uncertainty_overlay")
            if overlay is not None:
                App.markdown("**增强分歧热图（高亮 + 描边，越亮分歧越大）**")
                App.image(overlay, use_container_width=True)

            pairwise = explain_data.get("pairwise_iou", [])
            if pairwise:
                App.markdown("**模型两两IoU一致性**")
                App.dataframe(DataFrame(pairwise), hide_index=True, width="stretch")

            model_area = explain_data.get("model_area", [])
            if model_area:
                App.markdown("**各模型掩码面积占比**")
                App.dataframe(DataFrame(model_area), hide_index=True, width="stretch")

            App.info(_build_auto_summary(explain_data))
        else:
            col1, col2 = App.columns(2)
            with col1:
                App.metric("掩码面积占比", f"{explain_data.get('mask_area_ratio', 0.0) * 100:.2f}%")
            with col2:
                confidence = explain_data.get("avg_confidence")
                App.metric("平均置信度", "N/A" if confidence is None else f"{confidence:.4f}")

            model_name = explain_data.get("model_name", "未知模型")
            App.caption(f"解释来源模型：{model_name}")
            App.info(_build_auto_summary(explain_data))

    if use_expander:
        with App.expander("🧠 可解释分析", expanded=False):
            _render_body()
    else:
        _render_body()
