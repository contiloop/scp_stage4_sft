from __future__ import annotations

import unittest

from scp_stage4.config.loader import compose_config
from scp_stage4.config.validator import ConfigValidationError, validate_config


class ConfigValidationTests(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        validate_config(cfg)

    def test_backend_must_be_unsloth(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["training"]["backend"] = "hf_trainer"
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)

    def test_length_budget_must_fit_model(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["data"]["length"]["max_total_tokens"] = cfg["model"]["max_length"] + 1
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)

    def test_external_api_env_name_must_be_symbolic(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["external_api"]["primary"]["api_key_env"] = "sk-live-secret"
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)

    def test_logging_required_fields_must_include_config_hash(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["logging"]["required_event_fields"] = ["run_id", "subset_idx", "phase"]
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)

    def test_length_overflow_policy_must_be_supported(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["data"]["length"]["overflow"] = "invalid_mode"
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)

    def test_length_split_long_sentence_policy_must_be_supported(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["data"]["length"]["split"]["fallback_for_long_sentence"] = "invalid_mode"
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)

    def test_fixed_size_strategy_requires_fixed_size_key(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["pipeline"]["subset"]["strategy"] = "fixed_size"
        cfg["pipeline"]["subset"].pop("fixed_size", None)
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)

    def test_fixed_size_strategy_accepts_positive_integer(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["pipeline"]["subset"]["strategy"] = "fixed_size"
        cfg["pipeline"]["subset"]["fixed_size"] = 32
        validate_config(cfg)

    def test_subprocess_runtime_requires_command_list(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["inference"]["runtime"]["mode"] = "subprocess"
        cfg["inference"]["runtime"]["subprocess"]["command"] = None
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)

    def test_subprocess_runtime_accepts_command_lists(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["inference"]["runtime"]["mode"] = "subprocess"
        cfg["inference"]["runtime"]["subprocess"]["command"] = ["python", "-m", "x.y"]
        cfg["qe"]["runtime"]["mode"] = "subprocess"
        cfg["qe"]["runtime"]["subprocess"]["command"] = ["python", "-m", "x.y"]
        cfg["external_api"]["runtime"]["mode"] = "subprocess"
        cfg["external_api"]["runtime"]["subprocess"]["command"] = ["python", "-m", "x.y"]
        validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
