"""Unit tests for agent implementations."""

import shutil
import tempfile
import uuid
from pathlib import Path

from agentic_data_scientist.agents.adk.loop_detection import LoopDetectionAgent
from agentic_data_scientist.agents.claude_code.agent import (
    ClaudeCodeAgent,
    setup_skills_directory,
    setup_working_directory,
)


class TestClaudeCodeAgent:
    """Test ClaudeCodeAgent."""

    def test_initialization_default(self):
        """Test ClaudeCodeAgent default initialization."""
        agent = ClaudeCodeAgent()
        assert agent.name == "claude_coding_agent"
        assert agent.model == "claude-sonnet-4-6"
        assert agent._output_key == "implementation_summary"

    def test_initialization_custom(self):
        """Test ClaudeCodeAgent custom initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = ClaudeCodeAgent(
                name="custom_agent",
                description="Custom description",
                working_dir=tmpdir,
                output_key="custom_output",
            )
            assert agent.name == "custom_agent"
            assert agent.description == "Custom description"
            assert agent._working_dir == tmpdir
            assert agent._output_key == "custom_output"
            assert agent.model == "claude-sonnet-4-6"

    def test_truncate_summary_short(self):
        """Test summary truncation with short text."""
        agent = ClaudeCodeAgent()
        short_text = "Short summary"
        truncated = agent._truncate_summary(short_text)
        assert truncated == short_text

    def test_truncate_summary_long(self):
        """Test summary truncation with long text."""
        agent = ClaudeCodeAgent()
        long_text = "x" * 50000  # 50k characters
        truncated = agent._truncate_summary(long_text)
        assert len(truncated) <= 41000  # Should be around 40k + truncation message
        assert "middle section truncated" in truncated

    def test_should_retry_with_fallback_true_for_model_errors(self):
        """Fallback retry should trigger for model/provider-like failures."""
        agent = ClaudeCodeAgent(model="primary-model", fallback_model="backup-model")
        should_retry = agent._should_retry_with_fallback(Exception("Rate limit exceeded for model"))
        assert should_retry is True

    def test_should_retry_with_fallback_false_without_fallback(self):
        """Fallback retry should not trigger if no fallback model configured."""
        agent = ClaudeCodeAgent(model="primary-model")
        should_retry = agent._should_retry_with_fallback(Exception("Rate limit exceeded"))
        assert should_retry is False

    def test_should_retry_with_fallback_false_when_retry_disabled(self):
        """Fallback retry should not trigger when fallback_max_retries is 0."""
        agent = ClaudeCodeAgent(
            model="primary-model",
            fallback_model="backup-model",
            fallback_max_retries=0,
        )
        should_retry = agent._should_retry_with_fallback(Exception("Rate limit exceeded"))
        assert should_retry is False

    def test_should_retry_with_fallback_false_when_already_retrying(self):
        """Fallback retry should not loop recursively."""
        agent = ClaudeCodeAgent(model="primary-model", fallback_model="backup-model")
        agent._fallback_retrying = True
        should_retry = agent._should_retry_with_fallback(Exception("provider error"))
        assert should_retry is False


class TestSetupWorkingDirectory:
    """Test setup_working_directory function."""

    def test_create_directory_structure(self):
        """Test that working directory is created with proper structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            working_dir = Path(tmpdir) / "test_session"
            setup_working_directory(str(working_dir))

            assert working_dir.exists()
            assert (working_dir / "user_data").exists()
            assert (working_dir / "workflow").exists()
            assert (working_dir / "results").exists()
            assert (working_dir / "pyproject.toml").exists()
            assert (working_dir / "README.md").exists()

    def test_pyproject_content(self):
        """Test that pyproject.toml is created with proper content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            working_dir = Path(tmpdir) / "test_session"
            setup_working_directory(str(working_dir))

            pyproject_content = (working_dir / "pyproject.toml").read_text()
            assert "[project]" in pyproject_content
            assert "python" in pyproject_content.lower()

    def test_readme_content(self):
        """Test that README.md is created with proper content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            working_dir = Path(tmpdir) / "test_session"
            setup_working_directory(str(working_dir))

            readme_content = (working_dir / "README.md").read_text()
            assert "Agentic Data Scientist Session" in readme_content
            assert "user_data/" in readme_content
            assert "workflow/" in readme_content
            assert "results/" in readme_content

    def test_idempotent(self):
        """Test that setup is idempotent (can be called multiple times)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            working_dir = Path(tmpdir) / "test_session"

            # Call setup twice
            setup_working_directory(str(working_dir))
            setup_working_directory(str(working_dir))

            # Should still have correct structure
            assert (working_dir / "user_data").exists()
            assert (working_dir / "pyproject.toml").exists()


class TestSetupSkillsDirectory:
    """Test setup_skills_directory performance guards."""

    def _new_case_dir(self, prefix: str) -> Path:
        root = Path(".tmp") / "unit_skills_cases"
        root.mkdir(parents=True, exist_ok=True)
        case_dir = root / f"{prefix}_{uuid.uuid4().hex[:8]}"
        case_dir.mkdir(parents=True, exist_ok=True)
        return case_dir

    def test_reuses_existing_scoped_skills_without_overwrite(self):
        """If scoped skills already exist, setup should keep them and return early."""
        case_dir = self._new_case_dir("reuse_existing")
        try:
            existing_skill = case_dir / ".claude" / "skills" / "scientific-skills" / "existing_skill"
            existing_skill.mkdir(parents=True, exist_ok=True)
            (existing_skill / "SKILL.md").write_text("# existing", encoding="utf-8")

            # Even if a source path exists, early-return should keep existing skill untouched.
            source = case_dir / "scientific-skills"
            source.mkdir(parents=True, exist_ok=True)
            new_skill = source / "new_skill"
            new_skill.mkdir(parents=True, exist_ok=True)
            (new_skill / "SKILL.md").write_text("# new", encoding="utf-8")

            setup_skills_directory(str(case_dir))

            assert (existing_skill / "SKILL.md").exists()
            assert not (case_dir / ".claude" / "skills" / "scientific-skills" / "new_skill").exists()
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_copies_local_scientific_skills_source(self, monkeypatch):
        """Should copy skills from local scientific-skills source into scoped destination."""
        case_dir = self._new_case_dir("copy_local")
        try:
            source = case_dir / "scientific-skills"
            source.mkdir(parents=True, exist_ok=True)
            good_skill = source / "good_skill"
            good_skill.mkdir(parents=True, exist_ok=True)
            (good_skill / "SKILL.md").write_text("# good", encoding="utf-8")
            ignored = source / "ignored_dir"
            ignored.mkdir(parents=True, exist_ok=True)
            (ignored / "README.md").write_text("no skill marker", encoding="utf-8")

            monkeypatch.setenv("ADS_LOCAL_SKILLS_SOURCE", str(source))
            setup_skills_directory(str(case_dir))

            dst_root = case_dir / ".claude" / "skills" / "scientific-skills"
            assert (dst_root / "good_skill" / "SKILL.md").exists()
            assert not (dst_root / "ignored_dir").exists()
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)


class TestLoopDetectionAgentFallback:
    """Test one-shot fallback retry trigger logic for LoopDetectionAgent."""

    def test_should_retry_with_fallback_true_for_provider_errors(self):
        """Fallback should trigger for provider/model-like failures."""
        agent = LoopDetectionAgent(
            name="loop_agent",
            instruction="test",
            model="primary-model",
            fallback_model="backup-model",
        )
        should_retry = agent._should_retry_with_fallback(Exception("provider timeout while calling model"))
        assert should_retry is True

    def test_should_retry_with_fallback_false_without_fallback(self):
        """Fallback retry should not trigger if fallback model is absent."""
        agent = LoopDetectionAgent(name="loop_agent", instruction="test", model="primary-model")
        should_retry = agent._should_retry_with_fallback(Exception("rate limit exceeded"))
        assert should_retry is False

    def test_should_retry_with_fallback_false_when_retry_disabled(self):
        """Fallback retry should not trigger when fallback_max_retries is 0."""
        agent = LoopDetectionAgent(
            name="loop_agent",
            instruction="test",
            model="primary-model",
            fallback_model="backup-model",
            fallback_max_retries=0,
        )
        should_retry = agent._should_retry_with_fallback(Exception("provider timeout"))
        assert should_retry is False

    def test_should_retry_with_fallback_false_when_already_retrying(self):
        """Fallback retry should remain single-shot to avoid loops."""
        agent = LoopDetectionAgent(
            name="loop_agent",
            instruction="test",
            model="primary-model",
            fallback_model="backup-model",
        )
        agent._fallback_retrying = True
        should_retry = agent._should_retry_with_fallback(Exception("provider error"))
        assert should_retry is False

    def test_should_retry_with_fallback_false_when_models_are_same(self):
        """Fallback should not trigger when fallback resolves to same model identifier."""
        agent = LoopDetectionAgent(
            name="loop_agent",
            instruction="test",
            model="same-model",
            fallback_model="same-model",
        )
        should_retry = agent._should_retry_with_fallback(Exception("model not found"))
        assert should_retry is False
