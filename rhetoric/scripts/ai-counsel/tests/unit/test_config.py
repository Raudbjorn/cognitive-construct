"""Unit tests for configuration loading."""
import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from models.config import load_config


class TestConfigLoading:
    """Tests for config loading."""

    def test_load_default_config(self):
        """Test loading default config.yaml."""
        config = load_config()
        assert config is not None
        assert config.version == "2.0"
        assert "ollama" in config.adapters

    def test_adapter_config_structure(self):
        """Test adapter config has required fields."""
        config = load_config()
        ollama = config.adapters["ollama"]
        assert ollama.type == "http"
        assert ollama.base_url == "http://localhost:11434"
        assert ollama.timeout == 300

    def test_defaults_loaded(self):
        """Test default settings are loaded."""
        config = load_config()
        assert config.defaults.mode == "quick"
        assert config.defaults.rounds == 2
        assert config.defaults.max_rounds == 5

    def test_storage_config_loaded(self):
        """Test storage configuration is loaded."""
        config = load_config()
        assert config.storage.transcripts_dir == "transcripts"
        assert config.storage.format == "markdown"
        assert config.storage.auto_export is True

    def test_invalid_config_path_raises_error(self):
        """Test that invalid config path raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


class TestHTTPAdapterConfig:
    """Tests for HTTP adapter configuration."""

    def test_valid_http_adapter_config(self):
        """Test valid HTTP adapter configuration."""
        from models.config import HTTPAdapterConfig

        config = HTTPAdapterConfig(
            type="http", base_url="http://localhost:11434", timeout=60
        )
        assert config.type == "http"
        assert config.base_url == "http://localhost:11434"
        assert config.timeout == 60

    def test_http_adapter_with_api_key_env_var(self):
        """Test HTTP adapter with environment variable substitution."""
        from models.config import HTTPAdapterConfig

        os.environ["TEST_API_KEY"] = "sk-test-123"
        config = HTTPAdapterConfig(
            type="http",
            base_url="https://api.example.com",
            api_key="${TEST_API_KEY}",
            timeout=60,
        )
        # After loading, ${TEST_API_KEY} should be resolved
        assert config.api_key == "sk-test-123"
        del os.environ["TEST_API_KEY"]

    def test_http_adapter_requires_base_url(self):
        """Test that base_url field is required."""
        from models.config import HTTPAdapterConfig

        with pytest.raises(ValidationError):
            HTTPAdapterConfig(type="http", timeout=60)

    def test_http_adapter_missing_api_key_env_var_becomes_none(self):
        """Test that missing api_key environment variable gracefully becomes None."""
        from models.config import HTTPAdapterConfig

        # Make sure the env var doesn't exist
        if "NONEXISTENT_VAR" in os.environ:
            del os.environ["NONEXISTENT_VAR"]

        # api_key is optional, so missing env var should result in None
        config = HTTPAdapterConfig(
            type="http",
            base_url="http://test",
            api_key="${NONEXISTENT_VAR}",
            timeout=60,
        )
        assert config.api_key is None

    def test_http_adapter_missing_base_url_env_var_raises_error(self):
        """Test that missing required base_url env var raises clear error."""
        from models.config import HTTPAdapterConfig

        # Make sure the env var doesn't exist
        if "NONEXISTENT_BASE_URL_VAR" in os.environ:
            del os.environ["NONEXISTENT_BASE_URL_VAR"]

        # base_url is required, so missing env var should raise error
        with pytest.raises(ValidationError) as exc_info:
            HTTPAdapterConfig(
                type="http",
                base_url="${NONEXISTENT_BASE_URL_VAR}",
                timeout=60,
            )

        assert "NONEXISTENT_BASE_URL_VAR" in str(exc_info.value)


class TestConfigLoader:
    """Tests for config loader."""

    def test_load_config_with_adapters_section(self, tmp_path):
        """Test loading config with adapters section."""
        config_data = {
            "version": "2.0",
            "adapters": {
                "ollama": {
                    "type": "http",
                    "base_url": "http://localhost:11434",
                    "timeout": 60,
                }
            },
            "defaults": {
                "mode": "quick",
                "rounds": 2,
                "max_rounds": 5,
                "timeout_per_round": 120,
            },
            "storage": {
                "transcripts_dir": "transcripts",
                "format": "markdown",
                "auto_export": True,
            },
            "deliberation": {
                "convergence_detection": {
                    "enabled": True,
                    "semantic_similarity_threshold": 0.85,
                    "divergence_threshold": 0.40,
                    "min_rounds_before_check": 1,
                    "consecutive_stable_rounds": 2,
                    "stance_stability_threshold": 0.80,
                    "response_length_drop_threshold": 0.40,
                },
                "early_stopping": {
                    "enabled": True,
                    "threshold": 0.66,
                    "respect_min_rounds": True,
                },
                "convergence_threshold": 0.8,
                "enable_convergence_detection": True,
            },
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_file))
        assert config.adapters is not None
        assert "ollama" in config.adapters

    def test_load_config_fails_without_adapter_section(self, tmp_path):
        """Test config without adapters raises error."""
        config_data = {
            "version": "2.0",
            # Missing adapters section
            "defaults": {
                "mode": "quick",
                "rounds": 2,
                "max_rounds": 5,
                "timeout_per_round": 120,
            },
            "storage": {
                "transcripts_dir": "transcripts",
                "format": "markdown",
                "auto_export": True,
            },
            "deliberation": {
                "convergence_detection": {
                    "enabled": True,
                    "semantic_similarity_threshold": 0.85,
                    "divergence_threshold": 0.40,
                    "min_rounds_before_check": 1,
                    "consecutive_stable_rounds": 2,
                    "stance_stability_threshold": 0.80,
                    "response_length_drop_threshold": 0.40,
                },
                "early_stopping": {
                    "enabled": True,
                    "threshold": 0.66,
                    "respect_min_rounds": True,
                },
                "convergence_threshold": 0.8,
                "enable_convergence_detection": True,
            },
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValueError) as exc_info:
            load_config(str(config_file))

        assert "adapters" in str(exc_info.value).lower()


class TestDecisionGraphConfig:
    """Tests for DecisionGraphConfig path resolution."""

    @pytest.fixture
    def project_root(self):
        """
        Get the actual project root directory.

        The project root is where config.yaml is located, which is two levels
        up from models/config.py where DecisionGraphConfig is defined.
        """
        # This mirrors the logic in DecisionGraphConfig.resolve_db_path
        config_module_path = Path(__file__).parent.parent.parent / "models" / "config.py"
        return config_module_path.parent.parent

    def test_db_path_relative_to_project_root(self, project_root):
        """Test that relative path is resolved relative to project root."""
        from models.config import DecisionGraphConfig

        config = DecisionGraphConfig(enabled=True, db_path="decision_graph.db")

        # Path should be absolute
        resolved_path = Path(config.db_path)
        assert resolved_path.is_absolute(), "Resolved path should be absolute"

        # Path should be at project root
        expected_path = (project_root / "decision_graph.db").resolve()
        assert config.db_path == str(expected_path), (
            f"Expected {expected_path}, got {config.db_path}"
        )

    def test_db_path_absolute_unchanged(self, project_root):
        """Test that absolute paths are kept unchanged."""
        from models.config import DecisionGraphConfig

        absolute_path = "/tmp/test_graph.db"
        config = DecisionGraphConfig(enabled=True, db_path=absolute_path)

        resolved_path = Path(config.db_path)
        assert resolved_path.is_absolute(), "Absolute path should remain absolute"
        assert config.db_path == absolute_path

    def test_db_path_with_env_var(self, project_root, monkeypatch):
        """Test that environment variables are resolved before path resolution."""
        from models.config import DecisionGraphConfig

        test_data_dir = "/var/data"
        monkeypatch.setenv("TEST_DATA_DIR", test_data_dir)

        config = DecisionGraphConfig(
            enabled=True,
            db_path="${TEST_DATA_DIR}/graph.db"
        )

        expected_path = "/var/data/graph.db"
        assert config.db_path == expected_path

    def test_db_path_with_relative_env_var(self, project_root, monkeypatch):
        """Test that relative paths in env vars are resolved relative to project root."""
        from models.config import DecisionGraphConfig

        monkeypatch.setenv("TEST_DATA_DIR", "data")

        config = DecisionGraphConfig(
            enabled=True,
            db_path="${TEST_DATA_DIR}/graph.db"
        )

        expected_path = (project_root / "data" / "graph.db").resolve()
        assert config.db_path == str(expected_path)

    def test_db_path_parent_directory(self, project_root):
        """Test that parent directory references are resolved correctly."""
        from models.config import DecisionGraphConfig

        config = DecisionGraphConfig(enabled=True, db_path="../shared/graph.db")

        expected_path = (project_root / ".." / "shared" / "graph.db").resolve()
        assert config.db_path == str(expected_path)

    def test_db_path_subdirectory(self, project_root):
        """Test that subdirectory paths preserve structure under project root."""
        from models.config import DecisionGraphConfig

        config = DecisionGraphConfig(
            enabled=True,
            db_path="data/graphs/decision_graph.db"
        )

        expected_path = (project_root / "data" / "graphs" / "decision_graph.db").resolve()
        assert config.db_path == str(expected_path)

    def test_db_path_missing_env_var(self):
        """Test that missing environment variables raise clear error."""
        from models.config import DecisionGraphConfig

        if "NONEXISTENT_TEST_VAR" in os.environ:
            del os.environ["NONEXISTENT_TEST_VAR"]

        with pytest.raises(ValidationError) as exc_info:
            DecisionGraphConfig(
                enabled=True,
                db_path="${NONEXISTENT_TEST_VAR}/graph.db"
            )

        error_message = str(exc_info.value)
        assert "NONEXISTENT_TEST_VAR" in error_message

    def test_db_path_multiple_env_vars(self, monkeypatch):
        """Test that multiple environment variable references are resolved."""
        from models.config import DecisionGraphConfig

        monkeypatch.setenv("TEST_BASE_DIR", "/opt/app")
        monkeypatch.setenv("TEST_DB_NAME", "decisions")

        config = DecisionGraphConfig(
            enabled=True,
            db_path="${TEST_BASE_DIR}/${TEST_DB_NAME}.db"
        )

        expected_path = Path("/opt/app/decisions.db").resolve()
        assert config.db_path == str(expected_path)

    def test_db_path_default_value(self, project_root):
        """Test that default db_path value is set correctly."""
        from models.config import DecisionGraphConfig

        config = DecisionGraphConfig(enabled=True)
        assert config.db_path == "decision_graph.db"

    def test_db_path_cwd_independence(self, project_root, tmp_path, monkeypatch):
        """Test that db_path resolution is independent of current working directory."""
        from models.config import DecisionGraphConfig

        monkeypatch.chdir(tmp_path)
        assert Path.cwd() != project_root

        config = DecisionGraphConfig(enabled=True, db_path="decision_graph.db")

        expected_path = (project_root / "decision_graph.db").resolve()
        assert config.db_path == str(expected_path)
        assert not config.db_path.startswith(str(tmp_path))

    def test_db_path_home_directory_expansion(self, project_root):
        """Test that home directory (~) references are treated as relative paths."""
        from models.config import DecisionGraphConfig

        config = DecisionGraphConfig(enabled=True, db_path="~/data/graph.db")

        expected_path = (project_root / "~" / "data" / "graph.db").resolve()
        assert config.db_path == str(expected_path)

    def test_db_path_validation_fields(self):
        """Test that other DecisionGraphConfig fields validate correctly."""
        from models.config import DecisionGraphConfig

        config = DecisionGraphConfig(
            enabled=True,
            db_path="test.db",
            similarity_threshold=0.8,
            max_context_decisions=5,
            compute_similarities=False,
        )

        assert config.enabled is True
        assert Path(config.db_path).is_absolute()
        assert config.similarity_threshold == 0.8
        assert config.max_context_decisions == 5
        assert config.compute_similarities is False

    def test_db_path_invalid_similarity_threshold_still_validates_path(self, project_root):
        """Test that db_path is validated even if other field validation fails."""
        from models.config import DecisionGraphConfig

        with pytest.raises(ValidationError) as exc_info:
            DecisionGraphConfig(
                enabled=True,
                db_path="test.db",
                similarity_threshold=1.5,
            )

        error_message = str(exc_info.value)
        assert "similarity_threshold" in error_message.lower()


class TestDecisionGraphBudgetAwareConfig:
    """Tests for budget-aware context injection configuration."""

    def test_decision_graph_config_budget_fields(self):
        """Budget fields exist with sensible defaults."""
        from models.config import DecisionGraphConfig

        config = DecisionGraphConfig(enabled=True)

        assert hasattr(config, 'context_token_budget')
        assert hasattr(config, 'tier_boundaries')
        assert hasattr(config, 'query_window')

        assert config.context_token_budget == 1500
        assert config.tier_boundaries == {"strong": 0.75, "moderate": 0.60}
        assert config.query_window == 1000

    def test_decision_graph_config_tier_boundaries_validation(self):
        """Tier boundaries must be in order: strong > moderate > 0."""
        from models.config import DecisionGraphConfig

        config = DecisionGraphConfig(
            enabled=True,
            tier_boundaries={"strong": 0.75, "moderate": 0.60}
        )
        assert config.tier_boundaries["strong"] > config.tier_boundaries["moderate"]

        with pytest.raises(ValidationError):
            DecisionGraphConfig(
                enabled=True,
                tier_boundaries={"strong": 0.60, "moderate": 0.60}
            )

        with pytest.raises(ValidationError):
            DecisionGraphConfig(
                enabled=True,
                tier_boundaries={"strong": 0.50, "moderate": 0.70}
            )

    def test_decision_graph_config_query_window_validation(self):
        """Query window must be >= 50 and <= 10000."""
        from models.config import DecisionGraphConfig

        config = DecisionGraphConfig(enabled=True, query_window=500)
        assert config.query_window == 500

        with pytest.raises(ValidationError):
            DecisionGraphConfig(enabled=True, query_window=49)

        with pytest.raises(ValidationError):
            DecisionGraphConfig(enabled=True, query_window=10001)

    def test_decision_graph_config_context_token_budget_validation(self):
        """Context token budget must be >= 500 and <= 10000."""
        from models.config import DecisionGraphConfig

        config = DecisionGraphConfig(enabled=True, context_token_budget=1500)
        assert config.context_token_budget == 1500

        with pytest.raises(ValidationError):
            DecisionGraphConfig(enabled=True, context_token_budget=499)

        with pytest.raises(ValidationError):
            DecisionGraphConfig(enabled=True, context_token_budget=10001)

    def test_decision_graph_config_backward_compatibility(self):
        """Old config (without new fields) still loads with defaults."""
        from models.config import DecisionGraphConfig

        config = DecisionGraphConfig(
            enabled=True,
            db_path="test.db",
            similarity_threshold=0.7,
            max_context_decisions=3,
            compute_similarities=True
        )

        assert config.enabled is True
        assert config.similarity_threshold == 0.7
        assert config.context_token_budget == 1500
        assert config.tier_boundaries == {"strong": 0.75, "moderate": 0.60}

    def test_config_yaml_loads_new_parameters(self):
        """Load config.yaml successfully with new budget-aware parameters."""
        config = load_config()

        assert config.decision_graph is not None
        assert hasattr(config.decision_graph, 'context_token_budget')
        assert hasattr(config.decision_graph, 'tier_boundaries')
        assert hasattr(config.decision_graph, 'query_window')

        assert config.decision_graph.context_token_budget == 1500
        assert config.decision_graph.tier_boundaries == {"strong": 0.75, "moderate": 0.60}
        assert config.decision_graph.query_window == 1000


class TestFileTreeConfig:
    """Tests for FileTreeConfig validation."""

    def test_file_tree_config_defaults(self):
        """Test FileTreeConfig default values."""
        from models.config import FileTreeConfig

        config = FileTreeConfig()
        assert config.max_depth == 3
        assert config.max_files == 100
        assert config.enabled is True

    def test_file_tree_config_custom_values(self):
        """Test FileTreeConfig with custom values."""
        from models.config import FileTreeConfig

        config = FileTreeConfig(max_depth=5, max_files=50, enabled=False)
        assert config.max_depth == 5
        assert config.max_files == 50
        assert config.enabled is False

    def test_file_tree_config_max_depth_validation(self):
        """Test FileTreeConfig validates max_depth range (1-10)."""
        from models.config import FileTreeConfig

        FileTreeConfig(max_depth=1)
        FileTreeConfig(max_depth=10)

        with pytest.raises(ValidationError):
            FileTreeConfig(max_depth=0)

        with pytest.raises(ValidationError):
            FileTreeConfig(max_depth=11)

    def test_file_tree_config_max_files_validation(self):
        """Test FileTreeConfig validates max_files range (10-1000)."""
        from models.config import FileTreeConfig

        FileTreeConfig(max_files=10)
        FileTreeConfig(max_files=1000)

        with pytest.raises(ValidationError):
            FileTreeConfig(max_files=5)

        with pytest.raises(ValidationError):
            FileTreeConfig(max_files=1001)

    def test_deliberation_config_has_file_tree(self):
        """Test DeliberationConfig includes file_tree field."""
        from models.config import DeliberationConfig, FileTreeConfig, ConvergenceDetectionConfig, EarlyStoppingConfig

        config = DeliberationConfig(
            convergence_detection=ConvergenceDetectionConfig(
                enabled=True,
                semantic_similarity_threshold=0.85,
                divergence_threshold=0.40,
                min_rounds_before_check=1,
                consecutive_stable_rounds=2,
                stance_stability_threshold=0.80,
                response_length_drop_threshold=0.40,
            ),
            early_stopping=EarlyStoppingConfig(
                enabled=True,
                threshold=0.66,
                respect_min_rounds=True,
            ),
            convergence_threshold=0.8,
            enable_convergence_detection=True,
        )

        assert hasattr(config, 'file_tree')
        assert isinstance(config.file_tree, FileTreeConfig)
        assert config.file_tree.max_depth == 3
        assert config.file_tree.max_files == 100

    def test_config_yaml_loads_file_tree(self):
        """Test config.yaml loads file_tree section successfully."""
        config = load_config()

        assert config.deliberation is not None
        assert hasattr(config.deliberation, 'file_tree')
        assert config.deliberation.file_tree.enabled is True
        assert config.deliberation.file_tree.max_depth == 3
        assert config.deliberation.file_tree.max_files == 100
