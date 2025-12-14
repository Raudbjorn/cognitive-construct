"""Mem0 API client implementation."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from .result import Err, Ok, Result
from .types import (
    ApiError,
    BatchDeleteResponse,
    BatchMemoryDelete,
    BatchMemoryUpdate,
    BatchUpdateResponse,
    EntitiesResponse,
    Entity,
    EntityType,
    ExportCreateResponse,
    ExportData,
    ExportSummary,
    FeedbackResponse,
    FeedbackValue,
    MemberRole,
    Memory,
    MemoryAddResponse,
    MemoryDeleteResponse,
    MemoryEvent,
    MemoryHistoryEntry,
    MemoryListResponse,
    MemoryResult,
    MemorySearchResponse,
    MemoryUpdateResponse,
    PingResponse,
    Project,
    ProjectCreateResponse,
    ProjectMember,
    ProjectMembersResponse,
    Webhook,
    WebhookDeleteResponse,
    WebhookListResponse,
)

# === Constants ===

DEFAULT_BASE_URL = "https://api.mem0.ai"
DEFAULT_TIMEOUT = 300.0
API_KEY_ENV_VAR = "MEM0_API_KEY"


# === Helper Functions ===


def _get_api_key(api_key: str | None = None) -> str | None:
    """Resolve API key from parameter or environment."""
    return api_key or os.environ.get(API_KEY_ENV_VAR)


def _parse_memory(data: dict[str, Any]) -> Memory:
    """Parse API response into Memory."""
    return Memory(
        id=data.get("id", ""),
        memory=data.get("memory", ""),
        hash=data.get("hash"),
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
        metadata=data.get("metadata", {}),
        score=data.get("score"),
        categories=data.get("categories", []),
    )


# === Client Class ===


@dataclass
class Mem0Client:
    """Async client for Mem0 Cloud API.

    Usage:
        client = Mem0Client(api_key="...")  # or set MEM0_API_KEY env var

        # Add memory
        result = await client.add("User likes Italian food", user_id="user123")
        if result.is_ok():
            for item in result.value.results:
                print(f"{item.event}: {item.memory}")

        # Search memories
        result = await client.search("food preferences", user_id="user123")
        if result.is_ok():
            for memory in result.value.results:
                print(f"{memory.memory} (score: {memory.score})")
    """

    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    org_id: str | None = None
    project_id: str | None = None
    _resolved_key: str | None = field(init=False, repr=False, default=None)
    _user_id_hash: str | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        """Initialize resolved API key and user hash."""
        self._resolved_key = _get_api_key(self.api_key)
        if self._resolved_key:
            self._user_id_hash = hashlib.md5(self._resolved_key.encode()).hexdigest()

    def _headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {
            "Authorization": f"Token {self._resolved_key}",
            "Content-Type": "application/json",
            "User-Agent": "openmemory/1.0.0",
        }
        if self._user_id_hash:
            headers["Mem0-User-ID"] = self._user_id_hash
        return headers

    def _check_api_key(self) -> Result[None, ApiError]:
        """Verify API key is configured."""
        if not self._resolved_key:
            return Err(
                ApiError(
                    message=f"API key required. Set {API_KEY_ENV_VAR} or pass api_key parameter.",
                    code="AUTH_REQUIRED",
                )
            )
        return Ok(None)

    def _prepare_params(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Prepare query parameters with org/project IDs."""
        result = params.copy() if params else {}
        if self.org_id and self.project_id:
            result["org_id"] = self.org_id
            result["project_id"] = self.project_id
        return {k: v for k, v in result.items() if v is not None}

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Result[dict[str, Any], ApiError]:
        """Make HTTP request with error handling."""
        key_check = self._check_api_key()
        if key_check.is_err():
            return key_check  # type: ignore

        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_body,
                    headers=self._headers(),
                )

                if not response.is_success:
                    try:
                        error_data = response.json()
                        message = (
                            error_data.get("detail")
                            or error_data.get("error")
                            or error_data.get("message")
                            or response.reason_phrase
                        )
                    except Exception:
                        message = response.reason_phrase or f"HTTP {response.status_code}"

                    return Err(
                        ApiError(
                            message=str(message),
                            status_code=response.status_code,
                            code=f"HTTP_{response.status_code}",
                        )
                    )

                return Ok(response.json())

        except httpx.TimeoutException:
            return Err(ApiError(message="Request timed out", code="TIMEOUT"))
        except httpx.RequestError as e:
            return Err(ApiError(message=f"Request failed: {e}", code="REQUEST_ERROR"))

    # === Validation ===

    async def ping(self) -> Result[PingResponse, ApiError]:
        """Validate API key and get org/project info."""
        result = await self._request("GET", "/v1/ping/", params=self._prepare_params())
        if result.is_err():
            return result  # type: ignore

        data = result.value
        if data.get("org_id") and data.get("project_id"):
            object.__setattr__(self, "org_id", data.get("org_id"))
            object.__setattr__(self, "project_id", data.get("project_id"))

        return Ok(
            PingResponse(
                org_id=data.get("org_id"),
                project_id=data.get("project_id"),
                user_email=data.get("user_email"),
            )
        )

    # === Memory CRUD ===

    async def add(
        self,
        messages: str | dict[str, str] | list[dict[str, str]],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
        async_mode: bool = True,
    ) -> Result[MemoryAddResponse, ApiError]:
        """Add a new memory.

        Args:
            messages: Content to add. Can be string, single message dict, or list.
            user_id: User identifier for the memory.
            agent_id: Agent identifier for the memory.
            app_id: App identifier for the memory.
            run_id: Run identifier for the memory.
            metadata: Additional metadata to store.
            filters: Filters for memory matching.
            async_mode: Whether to process asynchronously (default: True).

        Returns:
            Result containing MemoryAddResponse on success or ApiError on failure.
        """
        if isinstance(messages, str):
            normalized_messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            normalized_messages = [messages]
        else:
            normalized_messages = messages

        payload: dict[str, Any] = {
            "messages": normalized_messages,
            "async_mode": async_mode,
            "output_format": "v1.1",
        }
        if user_id:
            payload["user_id"] = user_id
        if agent_id:
            payload["agent_id"] = agent_id
        if app_id:
            payload["app_id"] = app_id
        if run_id:
            payload["run_id"] = run_id
        if metadata:
            payload["metadata"] = metadata
        if filters:
            payload["filters"] = filters
        payload.update(self._prepare_params())

        result = await self._request("POST", "/v1/memories/", json_body=payload)
        if result.is_err():
            return result  # type: ignore

        data = result.value
        results = [
            MemoryResult(
                id=r.get("id", ""),
                event=MemoryEvent(r.get("event", "ADD")),
                memory=r.get("memory", ""),
            )
            for r in data.get("results", [])
        ]
        return Ok(MemoryAddResponse(results=results))

    async def get(self, memory_id: str) -> Result[Memory, ApiError]:
        """Get a specific memory by ID.

        Args:
            memory_id: The ID of the memory to retrieve.

        Returns:
            Result containing Memory on success or ApiError on failure.
        """
        result = await self._request(
            "GET",
            f"/v1/memories/{memory_id}/",
            params=self._prepare_params(),
        )
        if result.is_err():
            return result  # type: ignore

        return Ok(_parse_memory(result.value))

    async def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        run_id: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
    ) -> Result[MemoryListResponse, ApiError]:
        """Get all memories with optional filtering.

        Args:
            user_id: Filter by user ID.
            agent_id: Filter by agent ID.
            app_id: Filter by app ID.
            run_id: Filter by run ID.
            page: Page number for pagination.
            page_size: Number of results per page.

        Returns:
            Result containing MemoryListResponse on success or ApiError on failure.
        """
        body_params = self._prepare_params(
            {
                "user_id": user_id,
                "agent_id": agent_id,
                "app_id": app_id,
                "run_id": run_id,
            }
        )

        query_params: dict[str, Any] = {}
        if page is not None:
            query_params["page"] = page
        if page_size is not None:
            query_params["page_size"] = page_size

        result = await self._request(
            "POST",
            "/v2/memories/",
            params=query_params or None,
            json_body=body_params,
        )
        if result.is_err():
            return result  # type: ignore

        data = result.value
        if isinstance(data, list):
            memories = [_parse_memory(m) for m in data]
            return Ok(MemoryListResponse(results=memories))

        memories = [_parse_memory(m) for m in data.get("results", [])]
        return Ok(
            MemoryListResponse(
                results=memories,
                page=data.get("page"),
                page_size=data.get("page_size"),
                total=data.get("total"),
            )
        )

    async def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        run_id: str | None = None,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> Result[MemorySearchResponse, ApiError]:
        """Search memories by semantic similarity.

        Args:
            query: The search query string.
            user_id: Filter by user ID.
            agent_id: Filter by agent ID.
            app_id: Filter by app ID.
            run_id: Filter by run ID.
            top_k: Maximum number of results to return.
            filters: Additional filters.

        Returns:
            Result containing MemorySearchResponse on success or ApiError on failure.
        """
        payload: dict[str, Any] = {"query": query}
        if user_id:
            payload["user_id"] = user_id
        if agent_id:
            payload["agent_id"] = agent_id
        if app_id:
            payload["app_id"] = app_id
        if run_id:
            payload["run_id"] = run_id
        if top_k:
            payload["top_k"] = top_k
        if filters:
            payload["filters"] = filters
        payload.update(self._prepare_params())

        result = await self._request("POST", "/v2/memories/search/", json_body=payload)
        if result.is_err():
            return result  # type: ignore

        data = result.value
        if isinstance(data, list):
            memories = [_parse_memory(m) for m in data]
        else:
            memories = [_parse_memory(m) for m in data.get("results", [])]

        return Ok(MemorySearchResponse(results=memories))

    async def update(
        self,
        memory_id: str,
        *,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Result[MemoryUpdateResponse, ApiError]:
        """Update a memory by ID.

        Args:
            memory_id: The ID of the memory to update.
            text: New text content for the memory.
            metadata: New metadata for the memory.

        Returns:
            Result containing MemoryUpdateResponse on success or ApiError on failure.
        """
        if text is None and metadata is None:
            return Err(
                ApiError(
                    message="Either text or metadata must be provided",
                    code="VALIDATION_ERROR",
                )
            )

        payload: dict[str, Any] = {}
        if text is not None:
            payload["text"] = text
        if metadata is not None:
            payload["metadata"] = metadata

        result = await self._request(
            "PUT",
            f"/v1/memories/{memory_id}/",
            params=self._prepare_params(),
            json_body=payload,
        )
        if result.is_err():
            return result  # type: ignore

        data = result.value
        return Ok(
            MemoryUpdateResponse(
                id=data.get("id", memory_id),
                memory=data.get("memory", ""),
                updated_at=data.get("updated_at"),
            )
        )

    async def delete(self, memory_id: str) -> Result[MemoryDeleteResponse, ApiError]:
        """Delete a specific memory by ID.

        Args:
            memory_id: The ID of the memory to delete.

        Returns:
            Result containing MemoryDeleteResponse on success or ApiError on failure.
        """
        result = await self._request(
            "DELETE",
            f"/v1/memories/{memory_id}/",
            params=self._prepare_params(),
        )
        if result.is_err():
            return result  # type: ignore

        return Ok(MemoryDeleteResponse(message=result.value.get("message", "Deleted")))

    async def delete_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
    ) -> Result[MemoryDeleteResponse, ApiError]:
        """Delete all memories with optional filtering.

        Args:
            user_id: Delete memories for this user.
            agent_id: Delete memories for this agent.
            app_id: Delete memories for this app.

        Returns:
            Result containing MemoryDeleteResponse on success or ApiError on failure.
        """
        params = self._prepare_params(
            {
                "user_id": user_id,
                "agent_id": agent_id,
                "app_id": app_id,
            }
        )

        result = await self._request("DELETE", "/v1/memories/", params=params)
        if result.is_err():
            return result  # type: ignore

        return Ok(MemoryDeleteResponse(message=result.value.get("message", "Deleted")))

    async def history(self, memory_id: str) -> Result[list[MemoryHistoryEntry], ApiError]:
        """Get the history of a specific memory.

        Args:
            memory_id: The ID of the memory.

        Returns:
            Result containing list of MemoryHistoryEntry on success or ApiError on failure.
        """
        result = await self._request(
            "GET",
            f"/v1/memories/{memory_id}/history/",
            params=self._prepare_params(),
        )
        if result.is_err():
            return result  # type: ignore

        entries = [
            MemoryHistoryEntry(
                id=e.get("id", ""),
                memory_id=e.get("memory_id", memory_id),
                old_memory=e.get("old_memory"),
                new_memory=e.get("new_memory"),
                event=e.get("event", ""),
                created_at=e.get("created_at", ""),
                is_deleted=e.get("is_deleted", False),
            )
            for e in result.value
        ]
        return Ok(entries)

    # === Batch Operations ===

    async def batch_update(
        self, memories: list[BatchMemoryUpdate]
    ) -> Result[BatchUpdateResponse, ApiError]:
        """Batch update multiple memories.

        Args:
            memories: List of BatchMemoryUpdate objects.

        Returns:
            Result containing BatchUpdateResponse on success or ApiError on failure.
        """
        payload = {
            "memories": [
                {"memory_id": m.memory_id, "text": m.text, "metadata": m.metadata}
                for m in memories
            ]
        }

        result = await self._request("PUT", "/v1/batch/", json_body=payload)
        if result.is_err():
            return result  # type: ignore

        data = result.value
        return Ok(
            BatchUpdateResponse(
                updated=data.get("updated", len(memories)),
                errors=data.get("errors", []),
            )
        )

    async def batch_delete(
        self, memories: list[BatchMemoryDelete]
    ) -> Result[BatchDeleteResponse, ApiError]:
        """Batch delete multiple memories.

        Args:
            memories: List of BatchMemoryDelete objects.

        Returns:
            Result containing BatchDeleteResponse on success or ApiError on failure.
        """
        payload = {"memories": [{"memory_id": m.memory_id} for m in memories]}

        result = await self._request("DELETE", "/v1/batch/", json_body=payload)
        if result.is_err():
            return result  # type: ignore

        data = result.value
        return Ok(
            BatchDeleteResponse(
                deleted=data.get("deleted", len(memories)),
                errors=data.get("errors", []),
            )
        )

    # === Entity Operations ===

    async def get_entities(self) -> Result[EntitiesResponse, ApiError]:
        """Get all entities (users, agents, apps, runs).

        Returns:
            Result containing EntitiesResponse on success or ApiError on failure.
        """
        result = await self._request(
            "GET",
            "/v1/entities/",
            params=self._prepare_params(),
        )
        if result.is_err():
            return result  # type: ignore

        entities = [
            Entity(
                name=e.get("name", ""),
                type=EntityType(e.get("type", "user")),
                memory_count=e.get("memory_count"),
                created_at=e.get("created_at"),
            )
            for e in result.value.get("results", [])
        ]
        return Ok(EntitiesResponse(results=entities))

    async def delete_entity(
        self,
        entity_type: EntityType,
        entity_name: str,
    ) -> Result[MemoryDeleteResponse, ApiError]:
        """Delete a specific entity and its memories.

        Args:
            entity_type: The type of entity (user, agent, app, run).
            entity_name: The name/ID of the entity.

        Returns:
            Result containing MemoryDeleteResponse on success or ApiError on failure.
        """
        result = await self._request(
            "DELETE",
            f"/v2/entities/{entity_type.value}/{entity_name}/",
            params=self._prepare_params(),
        )
        if result.is_err():
            return result  # type: ignore

        return Ok(MemoryDeleteResponse(message="Entity deleted successfully"))

    async def reset(self) -> Result[MemoryDeleteResponse, ApiError]:
        """Reset all data (delete all entities and memories).

        Returns:
            Result containing MemoryDeleteResponse on success or ApiError on failure.
        """
        entities_result = await self.get_entities()
        if entities_result.is_err():
            return entities_result  # type: ignore

        for entity in entities_result.value.results:
            delete_result = await self.delete_entity(entity.type, entity.name)
            if delete_result.is_err():
                return delete_result

        return Ok(
            MemoryDeleteResponse(
                message="Client reset successful. All users and memories deleted."
            )
        )

    # === Export Operations ===

    async def create_export(
        self,
        schema: str,
        *,
        user_id: str | None = None,
        run_id: str | None = None,
    ) -> Result[ExportCreateResponse, ApiError]:
        """Create a memory export.

        Args:
            schema: Export schema.
            user_id: Filter by user ID.
            run_id: Filter by run ID.

        Returns:
            Result containing ExportCreateResponse on success or ApiError on failure.
        """
        payload: dict[str, Any] = {"schema": schema}
        payload.update(self._prepare_params({"user_id": user_id, "run_id": run_id}))

        result = await self._request("POST", "/v1/exports/", json_body=payload)
        if result.is_err():
            return result  # type: ignore

        data = result.value
        return Ok(
            ExportCreateResponse(
                request_id=data.get("request_id", ""),
                message=data.get("message", ""),
                status=data.get("status"),
            )
        )

    async def get_export(
        self,
        *,
        user_id: str | None = None,
    ) -> Result[ExportData, ApiError]:
        """Get exported memory data.

        Args:
            user_id: Filter by user ID.

        Returns:
            Result containing ExportData on success or ApiError on failure.
        """
        payload = self._prepare_params({"user_id": user_id})

        result = await self._request("POST", "/v1/exports/get/", json_body=payload)
        if result.is_err():
            return result  # type: ignore

        data = result.value
        return Ok(ExportData(data=data, status=data.get("status")))

    async def get_summary(
        self,
        filters: dict[str, Any] | None = None,
    ) -> Result[ExportSummary, ApiError]:
        """Get export summary.

        Args:
            filters: Optional filters.

        Returns:
            Result containing ExportSummary on success or ApiError on failure.
        """
        payload = self._prepare_params({"filters": filters})

        result = await self._request("POST", "/v1/summary/", json_body=payload)
        if result.is_err():
            return result  # type: ignore

        data = result.value
        return Ok(
            ExportSummary(
                status=data.get("status", ""),
                summary=data.get("summary"),
            )
        )

    # === Webhook Operations ===

    async def get_webhooks(self, project_id: str) -> Result[WebhookListResponse, ApiError]:
        """Get webhooks for a project.

        Args:
            project_id: The project ID.

        Returns:
            Result containing WebhookListResponse on success or ApiError on failure.
        """
        result = await self._request("GET", f"api/v1/webhooks/projects/{project_id}/")
        if result.is_err():
            return result  # type: ignore

        webhooks = [
            Webhook(
                id=w.get("id", 0),
                url=w.get("url", ""),
                name=w.get("name", ""),
                event_types=w.get("event_types", []),
                project_id=w.get("project_id"),
                is_active=w.get("is_active", True),
                created_at=w.get("created_at"),
            )
            for w in result.value.get("results", [])
        ]
        return Ok(WebhookListResponse(results=webhooks))

    async def create_webhook(
        self,
        url: str,
        name: str,
        project_id: str,
        event_types: list[str],
    ) -> Result[Webhook, ApiError]:
        """Create a webhook.

        Args:
            url: Webhook URL.
            name: Webhook name.
            project_id: Project ID.
            event_types: List of event types to trigger the webhook.

        Returns:
            Result containing Webhook on success or ApiError on failure.
        """
        payload = {"url": url, "name": name, "event_types": event_types}

        result = await self._request(
            "POST",
            f"api/v1/webhooks/projects/{project_id}/",
            json_body=payload,
        )
        if result.is_err():
            return result  # type: ignore

        data = result.value
        return Ok(
            Webhook(
                id=data.get("id", 0),
                url=data.get("url", url),
                name=data.get("name", name),
                event_types=data.get("event_types", event_types),
                project_id=project_id,
            )
        )

    async def update_webhook(
        self,
        webhook_id: int,
        *,
        name: str | None = None,
        url: str | None = None,
        event_types: list[str] | None = None,
    ) -> Result[Webhook, ApiError]:
        """Update a webhook.

        Args:
            webhook_id: The webhook ID.
            name: New name (optional).
            url: New URL (optional).
            event_types: New event types (optional).

        Returns:
            Result containing Webhook on success or ApiError on failure.
        """
        payload = {
            k: v
            for k, v in {
                "name": name,
                "url": url,
                "event_types": event_types,
            }.items()
            if v is not None
        }

        result = await self._request(
            "PUT",
            f"api/v1/webhooks/{webhook_id}/",
            json_body=payload,
        )
        if result.is_err():
            return result  # type: ignore

        data = result.value
        return Ok(
            Webhook(
                id=data.get("id", webhook_id),
                url=data.get("url", ""),
                name=data.get("name", ""),
                event_types=data.get("event_types", []),
            )
        )

    async def delete_webhook(self, webhook_id: int) -> Result[WebhookDeleteResponse, ApiError]:
        """Delete a webhook.

        Args:
            webhook_id: The webhook ID.

        Returns:
            Result containing WebhookDeleteResponse on success or ApiError on failure.
        """
        result = await self._request("DELETE", f"api/v1/webhooks/{webhook_id}/")
        if result.is_err():
            return result  # type: ignore

        return Ok(WebhookDeleteResponse(message=result.value.get("message", "Deleted")))

    # === Project Operations ===

    async def get_project(
        self,
        fields: list[str] | None = None,
    ) -> Result[Project, ApiError]:
        """Get project details.

        Args:
            fields: Optional list of fields to return.

        Returns:
            Result containing Project on success or ApiError on failure.
        """
        if not (self.org_id and self.project_id):
            return Err(
                ApiError(
                    message="org_id and project_id must be set",
                    code="VALIDATION_ERROR",
                )
            )

        params = self._prepare_params({"fields": fields})

        result = await self._request(
            "GET",
            f"/api/v1/orgs/organizations/{self.org_id}/projects/{self.project_id}/",
            params=params,
        )
        if result.is_err():
            return result  # type: ignore

        data = result.value
        return Ok(
            Project(
                id=data.get("id", self.project_id or ""),
                name=data.get("name", ""),
                org_id=data.get("org_id", self.org_id or ""),
                custom_instructions=data.get("custom_instructions"),
                custom_categories=data.get("custom_categories"),
                retrieval_criteria=data.get("retrieval_criteria"),
                enable_graph=data.get("enable_graph", False),
                version=data.get("version"),
                created_at=data.get("created_at"),
            )
        )

    async def update_project(
        self,
        *,
        custom_instructions: str | None = None,
        custom_categories: list[str] | None = None,
        retrieval_criteria: list[dict[str, Any]] | None = None,
        enable_graph: bool | None = None,
        version: str | None = None,
    ) -> Result[Project, ApiError]:
        """Update project settings.

        Args:
            custom_instructions: Custom instructions for the project.
            custom_categories: Custom memory categories.
            retrieval_criteria: Retrieval criteria configuration.
            enable_graph: Enable/disable graph support.
            version: Project version.

        Returns:
            Result containing Project on success or ApiError on failure.
        """
        if not (self.org_id and self.project_id):
            return Err(
                ApiError(
                    message="org_id and project_id must be set",
                    code="VALIDATION_ERROR",
                )
            )

        if all(
            v is None
            for v in [
                custom_instructions,
                custom_categories,
                retrieval_criteria,
                enable_graph,
                version,
            ]
        ):
            return Err(
                ApiError(
                    message="At least one field must be provided",
                    code="VALIDATION_ERROR",
                )
            )

        payload = {
            k: v
            for k, v in {
                "custom_instructions": custom_instructions,
                "custom_categories": custom_categories,
                "retrieval_criteria": retrieval_criteria,
                "enable_graph": enable_graph,
                "version": version,
            }.items()
            if v is not None
        }

        result = await self._request(
            "PATCH",
            f"/api/v1/orgs/organizations/{self.org_id}/projects/{self.project_id}/",
            json_body=payload,
        )
        if result.is_err():
            return result  # type: ignore

        data = result.value
        return Ok(
            Project(
                id=data.get("id", self.project_id or ""),
                name=data.get("name", ""),
                org_id=data.get("org_id", self.org_id or ""),
                custom_instructions=data.get("custom_instructions"),
                custom_categories=data.get("custom_categories"),
                retrieval_criteria=data.get("retrieval_criteria"),
                enable_graph=data.get("enable_graph", False),
                version=data.get("version"),
            )
        )

    async def create_project(
        self,
        name: str,
        description: str | None = None,
    ) -> Result[ProjectCreateResponse, ApiError]:
        """Create a new project.

        Args:
            name: Project name.
            description: Optional project description.

        Returns:
            Result containing ProjectCreateResponse on success or ApiError on failure.
        """
        if not self.org_id:
            return Err(
                ApiError(
                    message="org_id must be set",
                    code="VALIDATION_ERROR",
                )
            )

        payload: dict[str, Any] = {"name": name}
        if description:
            payload["description"] = description

        result = await self._request(
            "POST",
            f"/api/v1/orgs/organizations/{self.org_id}/projects/",
            json_body=payload,
        )
        if result.is_err():
            return result  # type: ignore

        data = result.value
        return Ok(
            ProjectCreateResponse(
                id=data.get("id", ""),
                name=data.get("name", name),
                message=data.get("message"),
            )
        )

    async def delete_project(self) -> Result[MemoryDeleteResponse, ApiError]:
        """Delete the current project.

        Returns:
            Result containing MemoryDeleteResponse on success or ApiError on failure.
        """
        if not (self.org_id and self.project_id):
            return Err(
                ApiError(
                    message="org_id and project_id must be set",
                    code="VALIDATION_ERROR",
                )
            )

        result = await self._request(
            "DELETE",
            f"/api/v1/orgs/organizations/{self.org_id}/projects/{self.project_id}/",
        )
        if result.is_err():
            return result  # type: ignore

        return Ok(MemoryDeleteResponse(message="Project deleted"))

    # === Project Members ===

    async def get_project_members(self) -> Result[ProjectMembersResponse, ApiError]:
        """Get project members.

        Returns:
            Result containing ProjectMembersResponse on success or ApiError on failure.
        """
        if not (self.org_id and self.project_id):
            return Err(
                ApiError(
                    message="org_id and project_id must be set",
                    code="VALIDATION_ERROR",
                )
            )

        result = await self._request(
            "GET",
            f"/api/v1/orgs/organizations/{self.org_id}/projects/{self.project_id}/members/",
        )
        if result.is_err():
            return result  # type: ignore

        members = [
            ProjectMember(
                email=m.get("email", ""),
                role=MemberRole(m.get("role", "READER")),
            )
            for m in result.value.get("results", [])
        ]
        return Ok(ProjectMembersResponse(results=members))

    async def add_project_member(
        self,
        email: str,
        role: MemberRole = MemberRole.READER,
    ) -> Result[ProjectMember, ApiError]:
        """Add a member to the project.

        Args:
            email: Member email.
            role: Member role (default: READER).

        Returns:
            Result containing ProjectMember on success or ApiError on failure.
        """
        if not (self.org_id and self.project_id):
            return Err(
                ApiError(
                    message="org_id and project_id must be set",
                    code="VALIDATION_ERROR",
                )
            )

        payload = {"email": email, "role": role.value}

        result = await self._request(
            "POST",
            f"/api/v1/orgs/organizations/{self.org_id}/projects/{self.project_id}/members/",
            json_body=payload,
        )
        if result.is_err():
            return result  # type: ignore

        return Ok(ProjectMember(email=email, role=role))

    async def update_project_member(
        self,
        email: str,
        role: MemberRole,
    ) -> Result[ProjectMember, ApiError]:
        """Update a project member's role.

        Args:
            email: Member email.
            role: New role.

        Returns:
            Result containing ProjectMember on success or ApiError on failure.
        """
        if not (self.org_id and self.project_id):
            return Err(
                ApiError(
                    message="org_id and project_id must be set",
                    code="VALIDATION_ERROR",
                )
            )

        payload = {"email": email, "role": role.value}

        result = await self._request(
            "PUT",
            f"/api/v1/orgs/organizations/{self.org_id}/projects/{self.project_id}/members/",
            json_body=payload,
        )
        if result.is_err():
            return result  # type: ignore

        return Ok(ProjectMember(email=email, role=role))

    async def remove_project_member(
        self,
        email: str,
    ) -> Result[MemoryDeleteResponse, ApiError]:
        """Remove a member from the project.

        Args:
            email: Member email.

        Returns:
            Result containing MemoryDeleteResponse on success or ApiError on failure.
        """
        if not (self.org_id and self.project_id):
            return Err(
                ApiError(
                    message="org_id and project_id must be set",
                    code="VALIDATION_ERROR",
                )
            )

        result = await self._request(
            "DELETE",
            f"/api/v1/orgs/organizations/{self.org_id}/projects/{self.project_id}/members/",
            params={"email": email},
        )
        if result.is_err():
            return result  # type: ignore

        return Ok(MemoryDeleteResponse(message="Member removed"))

    # === Feedback ===

    async def feedback(
        self,
        memory_id: str,
        feedback: FeedbackValue | None = None,
        feedback_reason: str | None = None,
    ) -> Result[FeedbackResponse, ApiError]:
        """Submit feedback for a memory.

        Args:
            memory_id: The memory ID.
            feedback: Feedback value (POSITIVE, NEGATIVE, VERY_NEGATIVE).
            feedback_reason: Optional reason for the feedback.

        Returns:
            Result containing FeedbackResponse on success or ApiError on failure.
        """
        payload: dict[str, Any] = {"memory_id": memory_id}
        if feedback:
            payload["feedback"] = feedback.value
        if feedback_reason:
            payload["feedback_reason"] = feedback_reason

        result = await self._request("POST", "/v1/feedback/", json_body=payload)
        if result.is_err():
            return result  # type: ignore

        return Ok(FeedbackResponse(message=result.value.get("message", "Feedback recorded")))


# === Convenience Functions ===


async def add(
    messages: str | dict[str, str] | list[dict[str, str]],
    *,
    user_id: str | None = None,
    api_key: str | None = None,
) -> Result[MemoryAddResponse, ApiError]:
    """Add memory (convenience function).

    Creates a client for single use. For multiple requests,
    prefer creating a Mem0Client instance.
    """
    client = Mem0Client(api_key=api_key)
    return await client.add(messages, user_id=user_id)


async def search(
    query: str,
    *,
    user_id: str | None = None,
    api_key: str | None = None,
) -> Result[MemorySearchResponse, ApiError]:
    """Search memories (convenience function).

    Creates a client for single use. For multiple requests,
    prefer creating a Mem0Client instance.
    """
    client = Mem0Client(api_key=api_key)
    return await client.search(query, user_id=user_id)
