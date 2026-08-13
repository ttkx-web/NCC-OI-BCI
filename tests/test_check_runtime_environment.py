from __future__ import annotations

import scripts.check_runtime_environment as probe


class _FakeTensor:
    def __mul__(self, value: float) -> _FakeTensor:
        assert value == 2.0
        return self

    def sum(self) -> _FakeTensor:
        return self

    def item(self) -> float:
        return 6.0


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_name(index: int) -> str:
        assert index == 0
        return "Test GPU"

    @staticmethod
    def get_device_capability(index: int) -> tuple[int, int]:
        assert index == 0
        return (12, 0)


class _FakeTorch:
    __version__ = "2.test"

    class version:
        cuda = "13.0"

    cuda = _FakeCuda()

    @staticmethod
    def tensor(values: list[float], *, device: str) -> _FakeTensor:
        assert values == [1.0, 2.0]
        assert device == "cuda"
        return _FakeTensor()


class _NoCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _NoCudaTorch(_FakeTorch):
    cuda = _NoCuda()


def test_collect_environment_reports_cuda_metadata_without_identity(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        probe,
        "_distribution_version",
        lambda name: f"{name}-version",
    )

    report = probe.collect_environment(torch_module=_FakeTorch())

    assert report["torch_version"] == "2.test"
    assert report["torch_cuda_runtime"] == "13.0"
    assert report["cuda_available"] is True
    assert report["gpu_name"] == "Test GPU"
    assert report["compute_capability"] == [12, 0]
    assert report["cuda_tensor_smoke"] == "passed"
    assert report["cuda_tensor_smoke_value"] == 6.0
    assert not any("serial" in key.lower() for key in report)


def test_collect_environment_skips_tensor_smoke_without_cuda() -> None:
    report = probe.collect_environment(torch_module=_NoCudaTorch())

    assert report["cuda_available"] is False
    assert report["gpu_name"] is None
    assert report["compute_capability"] is None
    assert report["cuda_tensor_smoke"] == "not_run"
