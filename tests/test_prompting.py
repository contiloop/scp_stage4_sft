from __future__ import annotations

from scp_stage4.pipeline.prompting import (
    PromptConfigError,
    render_sft_text,
    render_teacher_user_prompt,
    render_translation_prompt,
    teacher_system_prompt,
    teacher_prompt_hash,
)


def test_translation_prompt_is_stable_for_same_row() -> None:
    prompts = {
        "translation": {
            "templates": [
                "T1::{src}",
                "T2::{src}",
                "T3::{src}",
            ],
            "template_seed": 42,
            "selection_seed_scope": "row_id",
        }
    }
    prompt_a, idx_a = render_translation_prompt(
        prompts=prompts,
        source="Hello",
        row_id="row_001",
        subset_idx=0,
        metadata={},
    )
    prompt_b, idx_b = render_translation_prompt(
        prompts=prompts,
        source="Hello",
        row_id="row_001",
        subset_idx=7,
        metadata={},
    )
    assert prompt_a == prompt_b
    assert idx_a == idx_b
    assert "Hello" in prompt_a


def test_render_sft_text_uses_config_templates() -> None:
    prompts = {
        "sft": {
            "instruction_template": "INST::{source}\n",
            "response_template": "RESP::\n",
        }
    }
    text = render_sft_text(
        prompts=prompts,
        source="src text",
        target="tgt text",
    )
    assert text == "INST::src text\nRESP::\ntgt text"


def test_teacher_prompt_renders_metadata_payload() -> None:
    prompts = {
        "teacher_correction": {
            "system_template": "sys",
            "user_template": "S={source}\nM={metadata_json}\nD={student}",
            "metadata": {
                "include": True,
                "render_format": "json",
                "allowed_fields": ["dataset", "text_role"],
            },
        }
    }
    row = {
        "dataset": "alwaysgood/reuter",
        "source": "Hello",
        "student": "안녕",
        "metadata": {
            "text_role": "body",
            "title": "ignored by allowlist",
        },
    }
    rendered = render_teacher_user_prompt(prompts=prompts, row=row)
    assert "S=Hello" in rendered
    assert "D=안녕" in rendered
    assert "\"dataset\":\"alwaysgood/reuter\"" in rendered
    assert "\"text_role\":\"body\"" in rendered
    assert "title" not in rendered


def test_teacher_prompt_hash_changes_when_template_changes() -> None:
    cfg_a = {"teacher_correction": {"version": "v1", "system_template": "A"}}
    cfg_b = {"teacher_correction": {"version": "v1", "system_template": "B"}}
    assert teacher_prompt_hash(cfg_a) != teacher_prompt_hash(cfg_b)


def test_teacher_prompt_requires_teacher_correction_config() -> None:
    try:
        teacher_system_prompt({})
    except PromptConfigError:
        return
    raise AssertionError("expected PromptConfigError when prompts.teacher_correction is missing")
