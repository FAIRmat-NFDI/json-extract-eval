"""Tests for GroqJudge construction. No network -- the groq SDK is faked."""

import sys
import types

import pytest

from struct_extract_eval.batch.llm_judge import GroqJudge


class _FakeGroqClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key


@pytest.fixture
def fake_groq_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a stand-in ``groq`` module so GroqJudge() never touches the SDK."""
    module = types.ModuleType("groq")
    module.Groq = _FakeGroqClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "groq", module)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    return module


def test_default_model_is_gpt_oss_120b(fake_groq_module: types.ModuleType) -> None:
    judge = GroqJudge()
    assert judge.model == "openai/gpt-oss-120b"


def test_explicit_model_overrides_default(fake_groq_module: types.ModuleType) -> None:
    judge = GroqJudge(model="llama-3.3-70b-versatile")
    assert judge.model == "llama-3.3-70b-versatile"
