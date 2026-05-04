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

    def test_subset_archive_format_must_be_supported(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["pipeline"]["stage"]["subset_archive"]["format"] = "zip"
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)

    def test_real_profile_config_is_valid(self) -> None:
        cfg = compose_config("configs/scp_stage4_real.yaml")
        validate_config(cfg)

    def test_real_profile_defaults_to_hf_data_runtime(self) -> None:
        cfg = compose_config("configs/scp_stage4_real.yaml")
        self.assertEqual(cfg["data"]["runtime"]["mode"], "hf")

    def test_lora_target_modules_accepts_string_shortcut(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["training"]["base_update"]["lora"]["target_modules"] = "all-linear"
        validate_config(cfg)

    def test_lora_target_modules_rejects_invalid_type(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["training"]["base_update"]["lora"]["target_modules"] = 123
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)

    def test_data_num_workers_must_be_positive_integer_or_null(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["data"]["num_workers"] = 0
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)
        cfg["data"]["num_workers"] = None
        validate_config(cfg)

    def test_hf_dataset_download_workers_must_be_positive_integer_or_null(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["data"]["runtime"]["hf"]["dataset_download_workers"] = 0
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)
        cfg["data"]["runtime"]["hf"]["dataset_download_workers"] = None
        validate_config(cfg)

    def test_tokenizer_batch_size_must_be_positive_integer(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["data"]["length"]["tokenizer_batch_size"] = 0
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)
        cfg["data"]["length"]["tokenizer_batch_size"] = 256
        validate_config(cfg)

    def test_prepare_data_intermediate_format_must_be_supported(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["data"]["runtime"]["prepare_data"]["intermediate_format"] = "csv"
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)
        cfg["data"]["runtime"]["prepare_data"]["intermediate_format"] = "parquet"
        validate_config(cfg)

    def test_prepare_data_parquet_row_group_size_must_be_positive_integer(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["data"]["runtime"]["prepare_data"]["parquet_row_group_size"] = 0
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)
        cfg["data"]["runtime"]["prepare_data"]["parquet_row_group_size"] = 2048
        validate_config(cfg)

    def test_prepare_data_progress_config_contract(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["data"]["runtime"]["prepare_data"]["progress_enabled"] = "yes"
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["data"]["runtime"]["prepare_data"]["progress_every_rows"] = 0
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["data"]["runtime"]["prepare_data"]["progress_every_seconds"] = 0
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["data"]["runtime"]["prepare_data"]["progress_enabled"] = False
        cfg["data"]["runtime"]["prepare_data"]["progress_every_rows"] = 50000
        cfg["data"]["runtime"]["prepare_data"]["progress_every_seconds"] = 5.0
        validate_config(cfg)

    def test_inference_unsloth_runtime_flags_must_be_boolean(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["inference"]["runtime"]["unsloth"]["enabled"] = "yes"
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)

        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["inference"]["runtime"]["unsloth"]["fallback_to_transformers"] = 1
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)

    def test_inference_throughput_batching_contract(self) -> None:
        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["inference"]["throughput"]["batching"]["strategy"] = "fixed"
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)

        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["inference"]["throughput"]["batching"]["max_batch_tokens"] = 0
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)

        cfg = compose_config("configs/scp_stage4.yaml")
        cfg["inference"]["throughput"]["batching"]["pad_to_multiple_of"] = 0
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
