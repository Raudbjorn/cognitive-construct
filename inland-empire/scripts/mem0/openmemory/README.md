# openmemory

Pure Python client library for Mem0 Cloud API.

## Features

- **Async-first**: Built on httpx for high-performance async operations
- **Type-safe**: Full type hints with frozen dataclasses
- **Error handling**: Result types (no exceptions for expected errors)
- **Minimal**: Single dependency (httpx)

## Installation

```bash
pip install mem0ai
```

## Quick Start

```python
import asyncio
from openmemory import Mem0Client

async def main():
    # Initialize client (or set MEM0_API_KEY env var)
    client = Mem0Client(api_key="your-api-key")

    # Add a memory
    result = await client.add(
        "User prefers dark mode and likes Python",
        user_id="user123"
    )
    if result.is_ok():
        for item in result.value.results:
            print(f"{item.event}: {item.memory}")
    else:
        print(f"Error: {result.error.message}")

    # Search memories
    result = await client.search("preferences", user_id="user123")
    if result.is_ok():
        for memory in result.value.results:
            print(f"{memory.memory} (score: {memory.score})")

asyncio.run(main())
```

## Convenience Functions

For one-off operations:

```python
from openmemory import add, search

# Add memory
result = await add("User likes coffee", user_id="user123")

# Search memories
result = await search("beverages", user_id="user123")
```

## Error Handling

All methods return `Result[T, ApiError]`:

```python
result = await client.add("Hello", user_id="user123")

if result.is_ok():
    # Access success value
    response = result.value
    print(response.results)
else:
    # Handle error
    error = result.error
    print(f"Error: {error.message}")
    print(f"Status: {error.status_code}")

# Or use unwrap_or for defaults
memories = result.unwrap_or(MemoryAddResponse(results=[]))
```

## API Reference

### Memory Operations

```python
# Add memory
await client.add(messages, user_id=..., agent_id=..., metadata=...)

# Get specific memory
await client.get(memory_id)

# Get all memories
await client.get_all(user_id=..., page=1, page_size=50)

# Search memories
await client.search(query, user_id=..., top_k=10)

# Update memory
await client.update(memory_id, text=..., metadata=...)

# Delete memory
await client.delete(memory_id)

# Delete all memories
await client.delete_all(user_id=...)

# Get memory history
await client.history(memory_id)
```

### Batch Operations

```python
from openmemory import BatchMemoryUpdate, BatchMemoryDelete

# Batch update
await client.batch_update([
    BatchMemoryUpdate(memory_id="id1", text="new text"),
    BatchMemoryUpdate(memory_id="id2", metadata={"key": "value"}),
])

# Batch delete
await client.batch_delete([
    BatchMemoryDelete(memory_id="id1"),
    BatchMemoryDelete(memory_id="id2"),
])
```

### Entity Operations

```python
from openmemory import EntityType

# Get all entities
await client.get_entities()

# Delete entity
await client.delete_entity(EntityType.USER, "user123")

# Reset all data
await client.reset()
```

### Export Operations

```python
# Create export
await client.create_export(schema="v1")

# Get export data
await client.get_export(user_id="user123")

# Get export summary
await client.get_summary()
```

### Webhook Operations

```python
# List webhooks
await client.get_webhooks(project_id="proj123")

# Create webhook
await client.create_webhook(
    url="https://example.com/webhook",
    name="My Webhook",
    project_id="proj123",
    event_types=["memory.created", "memory.updated"]
)

# Update webhook
await client.update_webhook(webhook_id=1, name="New Name")

# Delete webhook
await client.delete_webhook(webhook_id=1)
```

### Project Operations

```python
from openmemory import MemberRole

# Get project
await client.get_project()

# Update project
await client.update_project(
    custom_instructions="Be helpful",
    enable_graph=True
)

# Create project
await client.create_project(name="New Project")

# Delete project
await client.delete_project()

# Manage members
await client.get_project_members()
await client.add_project_member("user@example.com", MemberRole.READER)
await client.update_project_member("user@example.com", MemberRole.OWNER)
await client.remove_project_member("user@example.com")
```

### Feedback

```python
from openmemory import FeedbackValue

await client.feedback(
    memory_id="mem123",
    feedback=FeedbackValue.POSITIVE,
    feedback_reason="Very helpful"
)
```

## Configuration

```python
client = Mem0Client(
    api_key="your-key",       # Or set MEM0_API_KEY env var
    base_url="https://...",   # Custom API endpoint
    timeout=300.0,            # Request timeout in seconds
    org_id="org123",          # Organization ID
    project_id="proj123",     # Project ID
)
```

## Types

All types are immutable frozen dataclasses:

```python
from openmemory import (
    # Memory types
    Memory, MemoryResult, MemoryAddResponse,
    MemoryListResponse, MemorySearchResponse,
    MemoryUpdateResponse, MemoryDeleteResponse,
    MemoryHistoryEntry,

    # Entity types
    Entity, EntitiesResponse, EntityType,

    # Batch types
    BatchMemoryUpdate, BatchMemoryDelete,
    BatchUpdateResponse, BatchDeleteResponse,

    # Export types
    ExportCreateResponse, ExportData, ExportSummary,

    # Webhook types
    Webhook, WebhookListResponse, WebhookDeleteResponse,

    # Project types
    Project, ProjectCreateResponse,
    ProjectMember, ProjectMembersResponse, MemberRole,

    # Other
    ApiError, FeedbackValue, FeedbackResponse,
    MemoryEvent, PingResponse,

    # Result types
    Result, Ok, Err,
)
```

## License

Apache-2.0
