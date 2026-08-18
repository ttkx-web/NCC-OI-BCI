"""Safe, non-medical rule-based Chinese text for the demo UI."""

from __future__ import annotations


def make_interpretation(states: dict[str, float], motor_intent: dict[str, object], band_power: dict[str, float]) -> str:
    engagement = states["cognitive_engagement"]
    relaxation = states["neural_relaxation"]
    stability = states["attention_stability"]
    label = str(motor_intent["label_cn"])
    confidence = float(motor_intent["confidence"])
    if relaxation >= 68.0 and engagement < 55.0:
        state_text = "Alpha 活动相对突出，整体节律呈现较放松的展示状态"
    elif engagement >= 68.0:
        state_text = "当前神经活动呈现较高参与度，Beta 相关活动较为明显"
    else:
        state_text = "当前神经状态保持在平稳的中等参与水平"
    confidence_text = "较集中" if confidence >= 0.60 else "正在平滑过渡"
    stability_text = "节律稳定" if stability >= 65.0 else "节律存在自然波动"
    dominant_band = max(band_power, key=band_power.get)
    return f"{state_text}。运动意图当前偏向{label}（{confidence_text}），{stability_text}；{dominant_band.title()} 频段贡献相对更高。"
