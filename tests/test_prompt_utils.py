"""
Unit tests for tome_core.utils.prompt_utils.

Covers: get_prompt_by_mode, list_available_prompt_modes, add_custom_prompt_mode,
        build_layout_preservation_prompt, build_simple_text_extraction_prompt,
        build_structured_data_prompt.
"""

import pytest

from tome_core.utils.prompt_utils import (
    get_prompt_by_mode,
    list_available_prompt_modes,
    add_custom_prompt_mode,
    build_layout_preservation_prompt,
    build_simple_text_extraction_prompt,
    build_structured_data_prompt,
)


class TestBuiltinPrompts:
    """Tests for the built-in prompt builder functions."""

    def test_layout_prompt_not_empty(self):
        prompt = build_layout_preservation_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_simple_extraction_not_empty(self):
        prompt = build_simple_text_extraction_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 5

    def test_structured_data_not_empty(self):
        prompt = build_structured_data_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50


class TestGetPromptByMode:
    """Tests for get_prompt_by_mode."""

    def test_known_mode_returns_string(self):
        prompt = get_prompt_by_mode("prompt_simple_extraction")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_all_builtin_modes_work(self):
        for mode in list_available_prompt_modes():
            prompt = get_prompt_by_mode(mode)
            assert isinstance(prompt, str)
            assert len(prompt) > 0, f"Prompt '{mode}' is empty"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid prompt_mode"):
            get_prompt_by_mode("nonexistent_mode_xyz")

    def test_unknown_mode_lists_available(self):
        with pytest.raises(ValueError) as exc:
            get_prompt_by_mode("bad_mode")
        assert "Available modes" in str(exc.value)


class TestListAvailablePromptModes:
    """Tests for list_available_prompt_modes."""

    def test_returns_list(self):
        modes = list_available_prompt_modes()
        assert isinstance(modes, list)

    def test_has_builtin_modes(self):
        modes = list_available_prompt_modes()
        assert "prompt_layout_all_en" in modes
        assert "prompt_simple_extraction" in modes
        assert "prompt_structured_data" in modes

    def test_all_mode_keys_are_valid(self):
        """Every listed mode should be retrievable."""
        for mode in list_available_prompt_modes():
            assert isinstance(get_prompt_by_mode(mode), str)


class TestAddCustomPromptMode:
    """Tests for add_custom_prompt_mode."""

    def test_adds_new_mode(self):
        add_custom_prompt_mode("test_custom", "This is a custom prompt")
        assert "test_custom" in list_available_prompt_modes()
        assert get_prompt_by_mode("test_custom") == "This is a custom prompt"

    def test_overwrites_existing(self):
        add_custom_prompt_mode("prompt_simple_extraction", "overwritten")
        assert get_prompt_by_mode("prompt_simple_extraction") == "overwritten"
        # Restore by re-adding
        add_custom_prompt_mode(
            "prompt_simple_extraction",
            build_simple_text_extraction_prompt(),
        )

    def test_mode_name_and_text_are_strings(self):
        add_custom_prompt_mode("str_test", "prompt text")
        result = get_prompt_by_mode("str_test")
        assert isinstance(result, str)
