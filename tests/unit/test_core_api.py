"""Unit tests for core API."""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_data_scientist.core.api import DataScientist, FileInfo, Result, SessionConfig
from agentic_data_scientist.core.history_store import HistoryStore
from agentic_data_scientist.core.state_contracts import StateKeys


class TestSessionConfig:
    """Test SessionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = SessionConfig()
        assert config.agent_type == "adk"
        assert config.mcp_servers is None
        assert config.max_llm_calls == 1024

    def test_custom_config(self):
        """Test custom configuration."""
        config = SessionConfig(
            agent_type="claude_code",
            mcp_servers=["filesystem", "fetch"],
            max_llm_calls=512,
        )
        assert config.agent_type == "claude_code"
        assert config.mcp_servers == ["filesystem", "fetch"]
        assert config.max_llm_calls == 512


class TestFileInfo:
    """Test FileInfo dataclass."""

    def test_file_info_creation(self):
        """Test FileInfo creation."""
        file_info = FileInfo(name="test.txt", path="/path/to/test.txt", size_kb=1.5)
        assert file_info.name == "test.txt"
        assert file_info.path == "/path/to/test.txt"
        assert file_info.size_kb == 1.5


class TestResult:
    """Test Result dataclass."""

    def test_successful_result(self):
        """Test successful result."""
        result = Result(
            session_id="test_session",
            status="completed",
            response="Test response",
            files_created=["output.txt"],
            duration=1.5,
            events_count=10,
        )
        assert result.session_id == "test_session"
        assert result.status == "completed"
        assert result.response == "Test response"
        assert result.error is None
        assert len(result.files_created) == 1
        assert result.duration == 1.5
        assert result.events_count == 10

    def test_error_result(self):
        """Test error result."""
        result = Result(
            session_id="test_session",
            status="error",
            error="Test error",
            duration=0.5,
        )
        assert result.session_id == "test_session"
        assert result.status == "error"
        assert result.error == "Test error"
        assert result.response is None


class TestDataScientist:
    """Test DataScientist class."""

    def test_initialization_adk(self):
        """Test DataScientist initialization with ADK agent."""
        ds = DataScientist(agent_type="adk")
        assert ds.config.agent_type == "adk"
        assert ds.session_id.startswith("session_")
        assert ds.working_dir.exists()
        ds.cleanup()

    def test_initialization_claude_code(self):
        """Test DataScientist initialization with Claude Code agent."""
        ds = DataScientist(agent_type="claude_code")
        assert ds.config.agent_type == "claude_code"
        assert ds.session_id.startswith("session_")
        assert ds.working_dir.exists()
        ds.cleanup()

    def test_save_files_bytes(self, tmp_path):
        """Test saving files from bytes."""
        ds = DataScientist(agent_type="adk")
        ds.working_dir = tmp_path

        content = b"Test content"
        files = [("test.txt", content)]

        file_info_list = ds.save_files(files)

        assert len(file_info_list) == 1
        assert file_info_list[0].name == "test.txt"
        assert Path(file_info_list[0].path).exists()
        assert Path(file_info_list[0].path).read_bytes() == content
        ds.cleanup()

    def test_prepare_prompt_no_files(self):
        """Test prompt preparation without files."""
        ds = DataScientist(agent_type="adk")
        message = "Test message"
        prompt = ds.prepare_prompt(message)
        assert prompt == message
        ds.cleanup()

    def test_prepare_prompt_with_files(self):
        """Test prompt preparation with files."""
        ds = DataScientist(agent_type="adk")
        message = "Analyze these files"
        file_info = [FileInfo(name="data.csv", path="/tmp/data.csv", size_kb=10.5)]

        prompt = ds.prepare_prompt(message, file_info)

        assert "Analyze these files" in prompt
        assert "data.csv" in prompt
        assert "10.5 KB" in prompt
        assert "user_data" in prompt
        ds.cleanup()

    def test_context_manager(self):
        """Test DataScientist as context manager."""
        with DataScientist(agent_type="adk") as ds:
            assert ds.working_dir.exists()

        # Cleanup should have been called
        # Note: cleanup is best-effort, directory may still exist

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test DataScientist as async context manager."""
        async with DataScientist(agent_type="adk") as ds:
            assert ds.working_dir.exists()

        # Cleanup should have been called

    @pytest.mark.asyncio
    async def test_collect_responses_preserves_original_input_state(self):
        """_collect_responses should pass raw user input separately from rendered prompt."""

        class DummyRunner:
            def __init__(self):
                self.last_state_delta = None

            async def run_async(self, user_id, session_id, new_message, state_delta):
                self.last_state_delta = state_delta
                if False:
                    yield None

        ds = DataScientist(agent_type="adk")
        ds.working_dir = Path(".test_core_api_workdir")
        ds.working_dir.mkdir(parents=True, exist_ok=True)
        ds.runner = DummyRunner()

        await ds._collect_responses("raw user request", "rendered prompt with file context", datetime.now())

        assert ds.runner.last_state_delta[StateKeys.ORIGINAL_USER_INPUT] == "raw user request"
        assert ds.runner.last_state_delta[StateKeys.LATEST_USER_INPUT] == "raw user request"
        assert ds.runner.last_state_delta[StateKeys.RENDERED_PROMPT] == "rendered prompt with file context"
        ds.cleanup()

    @pytest.mark.asyncio
    async def test_collect_responses_persists_history_rows(self, tmp_working_dir):
        """_collect_responses should persist compact history records when store is enabled."""

        class DummyRunner:
            async def run_async(self, user_id, session_id, new_message, state_delta):
                del user_id, session_id, new_message, state_delta
                if False:
                    yield None

        class DummySessionService:
            async def get_session(self, app_name, user_id, session_id):
                del app_name, user_id, session_id
                return SimpleNamespace(
                    state={
                        StateKeys.HIGH_LEVEL_STAGES: [
                            {
                                "index": 0,
                                "stage_id": "s1",
                                "title": "Stage 1",
                                "status": "approved",
                                "execution_mode": "workflow",
                                "workflow_id": "demo.workflow",
                            }
                        ],
                        StateKeys.STAGE_IMPLEMENTATIONS: [
                            {
                                "stage_index": 0,
                                "attempt": 1,
                                "approved": True,
                                "implementation_summary": "ok",
                                "review_reason": "approved",
                            }
                        ],
                        StateKeys.IMPLEMENTATION_REVIEW_CONFIRMATION_DECISION: {
                            "exit": True,
                            "reason": "approved",
                        },
                    }
                )

        ds = DataScientist(agent_type="adk")
        ds.working_dir = tmp_working_dir
        ds.runner = DummyRunner()
        ds.session_service = DummySessionService()
        ds.app = SimpleNamespace(name="agentic-data-scientist")
        db_path = Path.cwd() / f".history_test_api_{uuid.uuid4().hex}.sqlite3"
        ds._history_store = HistoryStore(db_path)

        result = await ds._collect_responses("raw", "prompt", datetime.now(), run_id="run_test_001")
        assert result.status == "completed"
        assert result.run_id == "run_test_001"

        conn = sqlite3.connect(db_path)
        try:
            run_count = conn.execute("SELECT COUNT(*) FROM run_summary WHERE run_id='run_test_001'").fetchone()[0]
            stage_count = conn.execute("SELECT COUNT(*) FROM stage_outcome WHERE run_id='run_test_001'").fetchone()[0]
            decision_count = conn.execute("SELECT COUNT(*) FROM decision_trace WHERE run_id='run_test_001'").fetchone()[0]
        finally:
            conn.close()

        assert run_count == 1
        assert stage_count == 1
        assert decision_count == 1
        ds.cleanup()

    def test_build_planner_history_advice_disabled(self, monkeypatch):
        """Planner advice should be disabled by env switch."""

        class DummyHistoryStore:
            def build_planner_advice(self, *, user_request, k, recent_limit):
                del user_request, k, recent_limit
                return "should-not-be-used"

        ds = DataScientist(agent_type="adk")
        ds._history_store = DummyHistoryStore()
        monkeypatch.setenv("ADS_LEARNING_ADVICE_ENABLED", "false")

        advice = ds._build_planner_history_advice(user_message="analyze rnaseq data")
        assert advice == ""
        ds.cleanup()

    def test_build_planner_history_advice_from_store(self, monkeypatch):
        """Planner advice should be pulled from history store when enabled."""

        class DummyHistoryStore:
            def __init__(self):
                self.calls = 0

            def build_planner_advice(self, *, user_request, k, recent_limit):
                self.calls += 1
                assert "rnaseq" in user_request
                assert k == 2
                assert recent_limit == 50
                return "Historical Planning Signals (advice-only): ..."

        ds = DataScientist(agent_type="adk")
        store = DummyHistoryStore()
        ds._history_store = store
        monkeypatch.setenv("ADS_LEARNING_ADVICE_ENABLED", "true")
        monkeypatch.setenv("ADS_LEARNING_TOPK", "2")
        monkeypatch.setenv("ADS_LEARNING_RECENT_RUNS", "50")

        advice = ds._build_planner_history_advice(user_message="rnaseq differential expression")
        assert "Historical Planning Signals" in advice
        assert store.calls == 1
        ds.cleanup()

    def test_build_planner_history_signals_disabled(self, monkeypatch):
        """Structured planner signals should be disabled by env switch."""

        class DummyHistoryStore:
            def build_planner_signals(self, *, user_request, k, recent_limit):
                del user_request, k, recent_limit
                return {"hot": {"run_count": 10}}

        ds = DataScientist(agent_type="adk")
        ds._history_store = DummyHistoryStore()
        monkeypatch.setenv("ADS_LEARNING_ADVICE_ENABLED", "false")

        signals = ds._build_planner_history_signals(user_message="analyze rnaseq data")
        assert signals == {}
        ds.cleanup()

    def test_build_planner_history_signals_from_store(self, monkeypatch):
        """Structured planner signals should be pulled from history store when enabled."""

        class DummyHistoryStore:
            def __init__(self):
                self.calls = 0

            def build_planner_signals(self, *, user_request, k, recent_limit):
                self.calls += 1
                assert "rnaseq" in user_request
                assert k == 2
                assert recent_limit == 50
                return {"hot": {"run_count": 12}, "topk_similar_runs": []}

        ds = DataScientist(agent_type="adk")
        store = DummyHistoryStore()
        ds._history_store = store
        monkeypatch.setenv("ADS_LEARNING_ADVICE_ENABLED", "true")
        monkeypatch.setenv("ADS_LEARNING_TOPK", "2")
        monkeypatch.setenv("ADS_LEARNING_RECENT_RUNS", "50")

        signals = ds._build_planner_history_signals(user_message="rnaseq differential expression")
        assert signals.get("hot", {}).get("run_count") == 12
        assert store.calls == 1
        ds.cleanup()

    def test_build_planner_skill_advice_disabled(self, monkeypatch):
        """Planner skill advice should be disabled by env switch."""
        ds = DataScientist(agent_type="adk")
        monkeypatch.setenv("ADS_PLANNER_SKILL_ADVICE_ENABLED", "false")
        advice = ds._build_planner_skill_advice(user_message="rnaseq differential expression")
        assert advice == ""
        ds.cleanup()

    def test_build_planner_skill_advice_from_local_skills(self, monkeypatch):
        """Planner skill advice should return top-matched skills when available."""
        root = Path(".tmp") / f"unit_core_api_skills_{uuid.uuid4().hex[:8]}"
        source = root / "scientific-skills"
        skill_dir = source / "rna-seq-analysis"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            "# RNA-seq Analysis\nDifferential expression and quality control workflow.",
            encoding="utf-8",
        )

        ds = DataScientist(agent_type="adk")
        ds.working_dir = root
        monkeypatch.setenv("ADS_PLANNER_SKILL_ADVICE_ENABLED", "true")
        monkeypatch.setenv("ADS_PLANNER_SKILL_TOPK", "3")
        monkeypatch.setenv("ADS_LOCAL_SKILLS_SOURCE", str(source))

        advice = ds._build_planner_skill_advice(user_message="Need RNA-seq differential expression")
        assert "Top matched scientific skills" in advice
        assert "rna-seq-analysis" in advice
        ds.cleanup()
