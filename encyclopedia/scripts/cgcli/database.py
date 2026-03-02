"""SurrealDB embedded connection management."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .result import Err, Ok, Result
from ._types import CodeGraphError

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    """SurrealDB connection configuration.

    Supports embedded engines:
        - "mem://"             — in-memory (tests, ephemeral)
        - "surrealkv://path"   — persistent file-based
        - "rocksdb://path"     — persistent file-based (alt engine)
    """

    db_url: str

    @classmethod
    def default(cls) -> DatabaseConfig:
        env_url = os.environ.get("CGCLI_DB_URL")
        if env_url:
            return cls(db_url=env_url)

        data_dir = Path(
            os.environ.get(
                "CGCLI_DATA_DIR",
                Path.home() / ".local" / "share" / "cgcli",
            )
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        return cls(db_url=f"surrealkv://{data_dir / 'codegraph'}")

    @classmethod
    def memory(cls) -> DatabaseConfig:
        return cls(db_url="mem://")


class DatabaseManager:
    """SurrealDB embedded connection manager (singleton, async)."""

    _instance: DatabaseManager | None = None
    _db = None  # AsyncSurreal — typed loosely to avoid import at module level
    _config: DatabaseConfig | None = None

    def __new__(cls) -> DatabaseManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def connect(
        self, config: DatabaseConfig | None = None
    ) -> Result[None, CodeGraphError]:
        """Connect to SurrealDB embedded engine and apply schema."""
        if self._db is not None:
            return Ok(None)

        config = config or self._config or DatabaseConfig.default()
        self._config = config

        try:
            from surrealdb import AsyncSurreal

            self._db = AsyncSurreal(config.db_url)
            await self._db.connect()
            await self._db.use("codegraph", "main")
            await self._apply_schema()
            log.info("Connected to SurrealDB: %s", config.db_url)
            return Ok(None)

        except ImportError:
            return Err(
                CodeGraphError(
                    "surrealdb package not installed",
                    "Run: pip install surrealdb",
                )
            )
        except Exception as e:
            self._db = None
            return Err(CodeGraphError(f"Connection failed: {e}"))

    async def _apply_schema(self) -> None:
        from .schema import SCHEMA

        await self._db.query(SCHEMA)

    def get_db(self):
        """Get the AsyncSurreal instance (connect first)."""
        if self._db is None:
            raise RuntimeError("DatabaseManager not connected. Call await connect() first.")
        return self._db

    async def query(self, sql: str, params: dict | None = None):
        """Execute a SurrealQL query and return results."""
        db = self.get_db()
        if params:
            return await db.query(sql, params)
        return await db.query(sql)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            log.info("SurrealDB connection closed")

    def is_connected(self) -> bool:
        return self._db is not None

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None
        cls._db = None
        cls._config = None
