from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from scp_stage4.pipeline.io_utils import read_jsonl
from scp_stage4.pipeline.smoke_local import run_smoke


class SmokeLocalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = "test_smoke_local"
        self.run_root = Path("artifacts/runs") / self.run_id
        if self.run_root.exists():
            shutil.rmtree(self.run_root)

    def tearDown(self) -> None:
        if self.run_root.exists():
            shutil.rmtree(self.run_root)

    def test_smoke_flow_writes_expected_artifacts_and_contracts(self) -> None:
        summary = run_smoke(
            config_path="configs/scp_stage4.yaml",
            run_id_override=self.run_id,
            subset_size_override=32,
        )

        subset_root = self.run_root / "subsets" / "subset_000"
        required_files = [
            self.run_root / "effective_config.yaml",
            self.run_root / "config_hash.txt",
            self.run_root / "events.jsonl",
            self.run_root / "metrics.jsonl",
            self.run_root / "failures.jsonl",
            self.run_root / "smoke_summary.json",
            subset_root / "input.jsonl",
            subset_root / "q1.jsonl",
            subset_root / "q2.jsonl",
            subset_root / "scored.jsonl",
            subset_root / "selected.jsonl",
            subset_root / "api_requests.jsonl",
            subset_root / "api.jsonl",
            subset_root / "events.jsonl",
            subset_root / "metrics.jsonl",
            subset_root / "failures.jsonl",
            subset_root / "train_final" / "train_rows.jsonl",
        ]
        for path in required_files:
            self.assertTrue(path.exists(), f"missing artifact: {path}")

        effective_config_text = (self.run_root / "effective_config.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("model:", effective_config_text)
        self.assertFalse(effective_config_text.lstrip().startswith("{"))

        counts = summary["counts"]
        self.assertGreaterEqual(counts["input"], 1)
        self.assertLessEqual(counts["input"], 32)
        self.assertEqual(counts["q1"], counts["input"])
        self.assertEqual(counts["q2"], counts["input"])
        self.assertEqual(counts["scored"], counts["input"])
        self.assertGreaterEqual(counts["selected"], 1)
        self.assertEqual(counts["selected"], counts["api_requests"])
        self.assertEqual(counts["selected"], counts["api"])
        self.assertEqual(counts["selected"], counts["train"])

        input_rows = read_jsonl(subset_root / "input.jsonl")
        q1_rows = read_jsonl(subset_root / "q1.jsonl")
        q2_rows = read_jsonl(subset_root / "q2.jsonl")
        scored_rows = read_jsonl(subset_root / "scored.jsonl")
        selected_rows = read_jsonl(subset_root / "selected.jsonl")
        api_rows = read_jsonl(subset_root / "api.jsonl")

        input_ids = [row["id"] for row in input_rows]
        self.assertEqual([row["id"] for row in q1_rows], input_ids)
        self.assertEqual([row["id"] for row in q2_rows], input_ids)
        self.assertEqual([row["id"] for row in scored_rows], input_ids)

        selected_ids = [row["id"] for row in selected_rows]
        self.assertTrue(set(selected_ids).issubset(set(input_ids)))
        self.assertEqual([row["id"] for row in api_rows], selected_ids)

        summary_json = json.loads((self.run_root / "smoke_summary.json").read_text())
        self.assertEqual(summary_json["run_id"], self.run_id)

        run_metrics_lines = (self.run_root / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertGreaterEqual(len(run_metrics_lines), 1)
        first_metric = json.loads(run_metrics_lines[0])
        self.assertEqual(first_metric["phase"], "smoke-local")
        self.assertIn("subset/input_rows", first_metric["metrics"])

        run_failures_text = (self.run_root / "failures.jsonl").read_text(encoding="utf-8")
        self.assertEqual(run_failures_text.strip(), "")


if __name__ == "__main__":
    unittest.main()
