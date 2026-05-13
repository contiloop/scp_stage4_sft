from __future__ import annotations

from typing import Any

from scp_stage4.pipeline.workers.qe_worker import _score_rows


def test_score_rows_xcomet_uses_metric_specific_model_override(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def _fake_comet_scores(
        rows: list[dict[str, Any]],
        *,
        model_name: str,
        batch_size: int,
        include_reference: bool = False,
    ) -> list[float]:
        captured["rows"] = rows
        captured["model_name"] = model_name
        captured["batch_size"] = batch_size
        captured["include_reference"] = include_reference
        return [0.777]

    monkeypatch.setattr(
        "scp_stage4.pipeline.workers.qe_worker._comet_scores",
        _fake_comet_scores,
    )

    rows = [
        {
            "id": "req-0",
            "backend": "xcomet",
            "src": "hello",
            "mt": "안녕하세요",
            "ref": "안녕",
            "runtime_config": {
                "qe_primary": {
                    "backend": "comet_kiwi",
                    "model_name": "Unbabel/wmt23-cometkiwi-da-xl",
                    "batch_size": 64,
                },
                "metric_settings": {
                    "xcomet": {
                        "model_name": "Unbabel/wmt22-comet-da",
                        "batch_size": 12,
                    }
                },
            },
        }
    ]

    scores, resolved_model = _score_rows(rows)

    assert scores == [0.777]
    assert resolved_model == "Unbabel/wmt22-comet-da"
    assert captured["model_name"] == "Unbabel/wmt22-comet-da"
    assert captured["batch_size"] == 12
    assert captured["include_reference"] is True


def test_score_rows_comet_kiwi_keeps_primary_model(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def _fake_comet_scores(
        rows: list[dict[str, Any]],
        *,
        model_name: str,
        batch_size: int,
        include_reference: bool = False,
    ) -> list[float]:
        captured["model_name"] = model_name
        captured["batch_size"] = batch_size
        captured["include_reference"] = include_reference
        return [0.555]

    monkeypatch.setattr(
        "scp_stage4.pipeline.workers.qe_worker._comet_scores",
        _fake_comet_scores,
    )

    rows = [
        {
            "id": "req-1",
            "backend": "comet_kiwi",
            "src": "hello",
            "mt": "안녕하세요",
            "runtime_config": {
                "qe_primary": {
                    "backend": "comet_kiwi",
                    "model_name": "Unbabel/wmt23-cometkiwi-da-xl",
                    "batch_size": 64,
                },
                "metric_settings": {
                    "xcomet": {
                        "model_name": "Unbabel/wmt22-comet-da",
                        "batch_size": 12,
                    }
                },
            },
        }
    ]

    scores, resolved_model = _score_rows(rows)

    assert scores == [0.555]
    assert resolved_model == "Unbabel/wmt23-cometkiwi-da-xl"
    assert captured["model_name"] == "Unbabel/wmt23-cometkiwi-da-xl"
    assert captured["batch_size"] == 64
    assert captured["include_reference"] is False
