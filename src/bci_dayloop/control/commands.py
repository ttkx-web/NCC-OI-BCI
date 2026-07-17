from __future__ import annotations

DEFAULT_COMMAND_MAP = {
    "left_hand": "LEFT",
    "right_hand": "RIGHT",
    "feet": "FORWARD",
    "tongue": "STOP",
}


def command_for_prediction(
    prediction: str,
    confidence: float,
    threshold: float,
    command_map: dict[str, str] | None = None,
) -> str:
    if confidence < threshold:
        return "STOP"
    return (command_map or DEFAULT_COMMAND_MAP).get(prediction, "STOP")

