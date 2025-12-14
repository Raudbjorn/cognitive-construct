"""Neo4j database connection management."""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass

from .result import Err, Ok, Result
from .types import CodeGraphError


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration."""

    uri: str
    username: str
    password: str

    @classmethod
    def from_env(cls) -> Result["Neo4jConfig", CodeGraphError]:
        """Create config from environment variables."""
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")

        if not uri:
            return Err(CodeGraphError("NEO4J_URI environment variable not set"))
        if not password:
            return Err(CodeGraphError("NEO4J_PASSWORD environment variable not set"))

        return Ok(cls(uri=uri, username=username, password=password))

    def validate(self) -> Result[None, CodeGraphError]:
        """Validate configuration."""
        uri_pattern = r"^(neo4j|neo4j\+s|neo4j\+ssc|bolt|bolt\+s|bolt\+ssc)://[^:]+:\d+$"
        if not re.match(uri_pattern, self.uri):
            return Err(
                CodeGraphError(
                    "Invalid Neo4j URI format",
                    "Expected format: neo4j://host:port or bolt://host:port",
                )
            )

        if not self.username or not self.username.strip():
            return Err(CodeGraphError("Username cannot be empty"))

        if not self.password or not self.password.strip():
            return Err(CodeGraphError("Password cannot be empty"))

        return Ok(None)


class DatabaseManager:
    """Thread-safe Neo4j database connection manager (singleton)."""

    _instance = None
    _driver = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._config: Neo4jConfig | None = None
        self._initialized = True

    def configure(self, config: Neo4jConfig) -> Result[None, CodeGraphError]:
        """Configure the database connection."""
        validation = config.validate()
        if validation.is_err():
            return validation
        self._config = config
        return Ok(None)

    def connect(self) -> Result[None, CodeGraphError]:
        """Establish database connection."""
        if self._driver is not None:
            return Ok(None)

        if self._config is None:
            # Try loading from environment
            config_result = Neo4jConfig.from_env()
            if config_result.is_err():
                return config_result
            self._config = config_result.value

        try:
            from neo4j import GraphDatabase

            with self._lock:
                if self._driver is None:
                    self._driver = GraphDatabase.driver(
                        self._config.uri,
                        auth=(self._config.username, self._config.password),
                    )
                    # Test connection
                    with self._driver.session() as session:
                        session.run("RETURN 1").consume()

            return Ok(None)

        except ImportError:
            return Err(
                CodeGraphError(
                    "neo4j package not installed", "Run: pip install neo4j>=5.15.0"
                )
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "authentication" in error_msg or "unauthorized" in error_msg:
                return Err(CodeGraphError("Authentication failed", "Invalid username or password"))
            elif "serviceunAvailable" in error_msg or "failed to establish" in error_msg:
                return Err(CodeGraphError("Neo4j service not available", "Is Neo4j running?"))
            else:
                return Err(CodeGraphError(f"Connection failed: {e}"))

    def get_driver(self):
        """Get the Neo4j driver (connects if needed)."""
        if self._driver is None:
            result = self.connect()
            if result.is_err():
                raise ValueError(result.error.message)
        return self._driver

    def close(self) -> None:
        """Close the database connection."""
        if self._driver is not None:
            with self._lock:
                if self._driver is not None:
                    self._driver.close()
                    self._driver = None

    def is_connected(self) -> bool:
        """Check if connected to database."""
        if self._driver is None:
            return False
        try:
            with self._driver.session() as session:
                session.run("RETURN 1").consume()
            return True
        except Exception:
            return False
