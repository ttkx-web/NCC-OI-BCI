from __future__ import annotations


def runtime_available() -> bool:
    try:
        from bci_dayloop.inference.runtime_control import PipelineController  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def system_status(run_service: object | None = None) -> dict[str, object]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except (ImportError, OSError):
        cuda_available = False
    device: dict[str, object] = {"status": "disconnected", "source": None}
    runtime: dict[str, object] = {"state": "idle"}
    if run_service is not None:
        try:
            runs = run_service.list()  # type: ignore[attr-defined]
            active = next((item for item in runs if item.run_type == "live" and item.state.value in {"starting", "running", "stopping"}), None)
            if active is not None:
                runtime = {"state": active.state.value, "run_id": active.id, "model_id": active.model_id}
                record = run_service.get(active.id)  # type: ignore[attr-defined]
                health = next((event.get("payload", {}) for event in reversed(record.broker.history) if event.get("type") == "device_health"), {})
                if isinstance(health, dict):
                    device = {"status": "connected" if health.get("connected") else "disconnected", "source": "neuracle_jellyfish", "health": health}
        except (AttributeError, LookupError):
            pass
    return {
        "device": device,
        "runtime": runtime,
        "compute": {"cuda_available": cuda_available},
    }
