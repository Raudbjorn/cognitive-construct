"""cgcli — code graph CLI powered by SurrealDB embedded."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from .client import CodeGraphClient
from .database import DatabaseConfig
from .formatters import detect_format, format_output


def _run(coro):
    """Run an async coroutine from sync click context."""
    return asyncio.run(coro)


def _output(ctx: click.Context, data, columns: list[str] | None = None) -> None:
    """Format and print output according to context format setting."""
    fmt = ctx.obj.get("format", "table")
    click.echo(format_output(data, fmt, columns))


def _handle_result(ctx: click.Context, result, columns: list[str] | None = None) -> None:
    """Handle a Result type: print value on Ok, error message on Err."""
    if result.is_err():
        click.echo(f"Error: {result.error.message}", err=True)
        if result.error.details:
            click.echo(f"  {result.error.details}", err=True)
        ctx.exit(1)
    else:
        _output(ctx, result.value, columns)


@click.group()
@click.option(
    "--db",
    envvar="CGCLI_DB_URL",
    default=None,
    help="Database URL (default: surrealkv://~/.local/share/cgcli/codegraph)",
)
@click.option(
    "--format", "-f",
    "output_format",
    type=click.Choice(["table", "json", "toon"]),
    default=None,
    help="Output format (default: table for TTY, toon for pipe)",
)
@click.pass_context
def cli(ctx: click.Context, db: str | None, output_format: str | None) -> None:
    """cgcli — code graph analysis tool."""
    ctx.ensure_object(dict)
    ctx.obj["db_url"] = db
    ctx.obj["format"] = detect_format(output_format)


async def _get_client(ctx: click.Context) -> CodeGraphClient:
    """Create and connect a client from click context."""
    db_url = ctx.obj.get("db_url")
    client = CodeGraphClient(db_url=db_url)
    config = DatabaseConfig(db_url=db_url) if db_url else None
    result = await client.connect(config)
    if result.is_err():
        click.echo(f"Error: {result.error.message}", err=True)
        if result.error.details:
            click.echo(f"  {result.error.details}", err=True)
        ctx.exit(1)
    return client


# -------------------------------------------------------------------------
# Commands
# -------------------------------------------------------------------------


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--dependency", is_flag=True, help="Mark as dependency code")
@click.pass_context
def index(ctx: click.Context, path: str, dependency: bool) -> None:
    """Index a repository or directory."""
    async def _run_index():
        client = await _get_client(ctx)
        try:
            resolved = str(Path(path).resolve())
            click.echo(f"Indexing {resolved}...")
            result = await client.index_repository(resolved, as_dependency=dependency)
            if result.is_err():
                click.echo(f"Error: {result.error.message}", err=True)
                ctx.exit(1)
            else:
                click.echo(f"Indexed successfully: {resolved}")
        finally:
            await client.close()

    _run(_run_index())


@cli.command()
@click.argument("name")
@click.option(
    "--type", "-t", "node_type",
    type=click.Choice(["func", "class", "var", "all"]),
    default="func",
    help="Type of code element to find",
)
@click.option("--fuzzy", is_flag=True, help="Use BM25 fuzzy search")
@click.pass_context
def find(ctx: click.Context, name: str, node_type: str, fuzzy: bool) -> None:
    """Find code elements by name."""
    async def _run_find():
        client = await _get_client(ctx)
        try:
            if node_type == "func":
                result = await client.find_function(name, fuzzy=fuzzy)
                _handle_result(ctx, result, ["name", "file_path", "line_number", "args"])
            elif node_type == "class":
                result = await client.find_class(name, fuzzy=fuzzy)
                _handle_result(ctx, result, ["name", "file_path", "line_number", "bases"])
            elif node_type == "var":
                result = await client.find_variable(name)
                _handle_result(ctx, result, ["name", "file_path", "line_number", "value"])
            else:
                # Search all types
                funcs = await client.find_function(name, fuzzy=fuzzy)
                classes = await client.find_class(name, fuzzy=fuzzy)
                variables = await client.find_variable(name)

                all_results: list = []
                if funcs.is_ok():
                    all_results.extend(funcs.value)
                if classes.is_ok():
                    all_results.extend(classes.value)
                if variables.is_ok():
                    all_results.extend(variables.value)

                _output(ctx, all_results, ["name", "file_path", "line_number"])
        finally:
            await client.close()

    _run(_run_find())


@cli.command()
@click.argument("name")
@click.option(
    "--direction", "-d",
    type=click.Choice(["callers", "callees", "both"]),
    default="callers",
    help="Direction of call graph",
)
@click.option("--file", "file_path", default=None, help="Filter by file path")
@click.pass_context
def calls(ctx: click.Context, name: str, direction: str, file_path: str | None) -> None:
    """Show call relationships for a function."""
    async def _run_calls():
        client = await _get_client(ctx)
        try:
            columns = [
                "caller_name", "caller_file_path",
                "called_name", "called_file_path", "call_line_number",
            ]
            if direction in ("callers", "both"):
                result = await client.who_calls(name, file_path=file_path)
                if direction == "callers":
                    _handle_result(ctx, result, columns)
                    return
                callers = result.value if result.is_ok() else []

            if direction in ("callees", "both"):
                result = await client.what_calls(name, file_path=file_path)
                if direction == "callees":
                    _handle_result(ctx, result, columns)
                    return
                callees = result.value if result.is_ok() else []

            if direction == "both":
                all_calls = callers + callees  # type: ignore[possibly-undefined]
                _output(ctx, all_calls, columns)
        finally:
            await client.close()

    _run(_run_calls())


@cli.command("imports")
@click.argument("module")
@click.pass_context
def who_imports(ctx: click.Context, module: str) -> None:
    """Show what files import a module."""
    async def _run_imports():
        client = await _get_client(ctx)
        try:
            result = await client.who_imports(module)
            _handle_result(ctx, result, ["file_name", "file_path", "module_name", "alias"])
        finally:
            await client.close()

    _run(_run_imports())


@cli.command()
@click.argument("class_name")
@click.option("--file", "file_path", default=None, help="Filter by file path")
@click.pass_context
def hierarchy(ctx: click.Context, class_name: str, file_path: str | None) -> None:
    """Show class inheritance hierarchy."""
    async def _run_hierarchy():
        client = await _get_client(ctx)
        try:
            result = await client.class_hierarchy(class_name, file_path=file_path)
            _handle_result(ctx, result)
        finally:
            await client.close()

    _run(_run_hierarchy())


@cli.command("dead-code")
@click.option("--exclude-decorator", "-e", multiple=True, help="Exclude functions with decorator")
@click.pass_context
def dead_code(ctx: click.Context, exclude_decorator: tuple[str, ...]) -> None:
    """Find potentially unused functions."""
    async def _run_dead():
        client = await _get_client(ctx)
        try:
            result = await client.find_dead_code(
                exclude_decorators=list(exclude_decorator) if exclude_decorator else None
            )
            _handle_result(ctx, result, ["name", "file_path", "line_number"])
        finally:
            await client.close()

    _run(_run_dead())


@cli.command()
@click.option("--limit", "-n", default=10, help="Number of results")
@click.pass_context
def complexity(ctx: click.Context, limit: int) -> None:
    """Find the most complex functions."""
    async def _run_complex():
        client = await _get_client(ctx)
        try:
            result = await client.most_complex_functions(limit=limit)
            _handle_result(
                ctx, result,
                ["name", "file_path", "cyclomatic_complexity", "line_number"],
            )
        finally:
            await client.close()

    _run(_run_complex())


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Number of results")
@click.pass_context
def search(ctx: click.Context, query: str, limit: int) -> None:
    """Semantic vector search across indexed code."""
    async def _run_search():
        client = await _get_client(ctx)
        try:
            result = await client.vector_search(query, limit=limit)
            _handle_result(
                ctx, result,
                ["name", "file_path", "search_type", "relevance_score", "line_number"],
            )
        finally:
            await client.close()

    _run(_run_search())


@cli.command()
@click.pass_context
def repos(ctx: click.Context) -> None:
    """List indexed repositories."""
    async def _run_repos():
        client = await _get_client(ctx)
        try:
            result = await client.list_repositories()
            _handle_result(ctx, result, ["name", "path", "is_dependency"])
        finally:
            await client.close()

    _run(_run_repos())


@cli.command()
@click.argument("path")
@click.pass_context
def delete(ctx: click.Context, path: str) -> None:
    """Delete an indexed repository."""
    async def _run_delete():
        client = await _get_client(ctx)
        try:
            result = await client.delete_repository(path)
            if result.is_err():
                click.echo(f"Error: {result.error.message}", err=True)
                ctx.exit(1)
            else:
                click.echo(f"Deleted: {path}")
        finally:
            await client.close()

    _run(_run_delete())


if __name__ == "__main__":
    cli()
