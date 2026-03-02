"""Base classes and config for the data-driven language parser system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from ..debug_log import error_logger, warning_logger
from ..tree_sitter_manager import execute_query


# ---------------------------------------------------------------------------
# Protocols for hook callables
# ---------------------------------------------------------------------------

class FinderFn(Protocol):
    """Signature for _find_* hooks: (extractor, root_node) -> list[dict]."""

    def __call__(self, ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]: ...


class PreParseHook(Protocol):
    """Called before parsing. Returns (source_code, actual_path, cleanup_fn | None)."""

    def __call__(
        self, path: Path, parser: Any
    ) -> tuple[str, Path, Callable[[], None] | None]: ...


class PreScanFn(Protocol):
    """Signature for pre_scan_* functions used by graph_builder."""

    def __call__(self, files: list[Path], parser_wrapper: Any) -> dict[str, list[str]]: ...


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LanguageConfig:
    """Declarative config for a single language parser."""

    name: str
    queries: dict[str, str]

    # AST node types that constitute "parent context"
    context_types: tuple[str, ...] = ()

    # Cyclomatic complexity node types (None = don't compute)
    complexity_nodes: frozenset[str] | None = None

    # Docstring strategy: "first_string" | "prev_comment" | "jsdoc" | "none"
    docstring_strategy: str = "none"

    # Custom finder hooks — if None, the extractor uses its default implementation.
    # Each takes (extractor, root_node) -> list[dict].
    find_functions: FinderFn | None = None
    find_classes: FinderFn | None = None
    find_imports: FinderFn | None = None
    find_calls: FinderFn | None = None
    find_variables: FinderFn | None = None

    # Extra entity finders: key → result dict key, value → finder fn
    # e.g. {"interfaces": find_go_interfaces, "traits": find_rust_traits}
    find_extra: dict[str, FinderFn] = field(default_factory=dict)

    # Pre-parse hook (e.g. notebook conversion). Returns (source, real_path, cleanup).
    pre_parse_hook: PreParseHook | None = None

    # Pre-scan function for graph_builder._pre_scan_for_imports
    pre_scan_fn: PreScanFn | None = None

    # Pre-scan query string (alternative to pre_scan_fn for simple cases)
    pre_scan_query: str | None = None


# ---------------------------------------------------------------------------
# Generic extractor
# ---------------------------------------------------------------------------

class GenericExtractor:
    """Config-driven parser that replaces all per-language parser classes."""

    def __init__(self, config: LanguageConfig, wrapper: Any) -> None:
        self.config = config
        self.language_name = config.name
        self.language = wrapper.language
        self.parser = wrapper.parser
        self.index_source = False
        self.current_path: str = ""
        self._parsed_variables: list[dict[str, Any]] = []

    # -- Utility methods (shared by all languages) -------------------------

    def get_node_text(self, node: Any) -> str:
        """Decode a tree-sitter node's text."""
        return node.text.decode("utf-8")

    def get_parent_context(
        self, node: Any, types: tuple[str, ...] | None = None
    ) -> tuple[str | None, str | None, int | None]:
        """Walk up the AST to find the nearest enclosing context node."""
        ctx_types = types or self.config.context_types
        curr = node.parent
        while curr:
            if curr.type in ctx_types:
                name_node = curr.child_by_field_name("name")
                name = self.get_node_text(name_node) if name_node else None
                return name, curr.type, curr.start_point[0] + 1
            curr = curr.parent
        return None, None, None

    def calculate_complexity(self, node: Any) -> int | None:
        """Compute cyclomatic complexity if configured."""
        if self.config.complexity_nodes is None:
            return None
        count = 1
        nodes = self.config.complexity_nodes

        def traverse(n: Any) -> None:
            nonlocal count
            if n.type in nodes:
                count += 1
            for child in n.children:
                traverse(child)

        traverse(node)
        return count

    def get_docstring(self, node: Any, body_node: Any = None) -> str | None:
        """Extract a docstring using the configured strategy."""
        strategy = self.config.docstring_strategy

        if strategy == "first_string":
            # Python-style: first expression_statement > string in body
            import ast as _ast

            target = body_node or node
            if target and target.child_count > 0:
                first = target.children[0]
                if (
                    first.type == "expression_statement"
                    and first.child_count > 0
                    and first.children[0].type == "string"
                ):
                    try:
                        return _ast.literal_eval(self.get_node_text(first.children[0]))
                    except (ValueError, SyntaxError):
                        return self.get_node_text(first.children[0])
            return None

        if strategy == "prev_comment":
            # Go / Rust style: preceding // or /// comment
            prev = node.prev_sibling
            while prev and prev.type in ("comment", "\n", " "):
                if prev.type == "comment":
                    text = self.get_node_text(prev)
                    if text.startswith("//"):
                        return text.strip()
                prev = prev.prev_sibling
            return None

        if strategy == "jsdoc":
            # JS/TS style: preceding /** ... */ block comment
            prev = node.prev_sibling
            while prev and prev.type in ("comment", "\n", " "):
                if prev.type == "comment":
                    text = self.get_node_text(prev)
                    if text.startswith("/**") and text.endswith("*/"):
                        return text.strip()
                prev = prev.prev_sibling
            return None

        return None

    def run_query(self, query_str: str, root_node: Any) -> list[tuple[Any, str]]:
        """Execute a tree-sitter query and return (node, capture_name) pairs."""
        return list(execute_query(self.language, query_str, root_node))

    # -- Default finders (used when config.find_X is None) -----------------

    def _default_find_calls(self, root_node: Any) -> list[dict[str, Any]]:
        """Generic call extraction — works for most languages."""
        query_str = self.config.queries.get("calls")
        if not query_str:
            return []

        calls: list[dict[str, Any]] = []
        seen: set[str] = set()

        for node, capture_name in self.run_query(query_str, root_node):
            if capture_name != "name":
                continue

            call_node = node.parent
            while call_node and call_node.type not in (
                "call_expression",
                "new_expression",
                "method_invocation",
                "invocation_expression",
                "object_creation_expression",
                "program",
                "source_file",
            ):
                call_node = call_node.parent

            name = self.get_node_text(node)
            line_number = node.start_point[0] + 1

            key = f"{name}_{line_number}"
            if key in seen:
                continue
            seen.add(key)

            # Try to get full call text
            full_name = name
            if call_node and call_node.type not in ("program", "source_file"):
                func_node = call_node.child_by_field_name("function")
                if func_node:
                    full_name = self.get_node_text(func_node)
                else:
                    full_name = self.get_node_text(call_node)

            # Extract arguments
            args: list[str] = []
            if call_node and call_node.type not in ("program", "source_file"):
                args_node = call_node.child_by_field_name("arguments")
                if args_node:
                    for arg in args_node.children:
                        if arg.type not in ("(", ")", ","):
                            args.append(self.get_node_text(arg))

            ctx_name, ctx_type, ctx_line = self.get_parent_context(node)

            calls.append(
                {
                    "name": name,
                    "full_name": full_name,
                    "line_number": line_number,
                    "args": args,
                    "inferred_obj_type": None,
                    "context": (ctx_name, ctx_type, ctx_line),
                    "class_context": None,
                    "lang": self.language_name,
                    "is_dependency": False,
                }
            )

        return calls

    def _default_find_variables(self, root_node: Any) -> list[dict[str, Any]]:
        """Generic variable extraction — works for most languages."""
        query_str = self.config.queries.get("variables")
        if not query_str:
            return []

        variables: list[dict[str, Any]] = []

        for node, capture_name in self.run_query(query_str, root_node):
            if capture_name != "name":
                continue

            name = self.get_node_text(node)
            ctx_name, _, _ = self.get_parent_context(node)

            variables.append(
                {
                    "name": name,
                    "line_number": node.start_point[0] + 1,
                    "value": None,
                    "type": None,
                    "context": ctx_name,
                    "class_context": None,
                    "lang": self.language_name,
                    "is_dependency": False,
                }
            )

        return variables

    # -- Main parse method -------------------------------------------------

    def parse(
        self,
        path: Path,
        is_dependency: bool = False,
        index_source: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Parse a file and return its structure as a standardized dict."""
        self.index_source = index_source
        original_path = path
        self.current_path = str(path)
        cleanup_fn: Callable[[], None] | None = None

        try:
            # Pre-parse hook (e.g. notebook → temp .py)
            if self.config.pre_parse_hook:
                source_code, path, cleanup_fn = self.config.pre_parse_hook(
                    path, self.parser, **kwargs
                )
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    source_code = f.read()

            if not source_code.strip():
                warning_logger(f"Empty or whitespace-only file: {path}")
                return self._empty_result(str(original_path), is_dependency)

            tree = self.parser.parse(bytes(source_code, "utf8"))
            root_node = tree.root_node

            cfg = self.config

            # Dispatch to hooks or defaults
            functions = (
                cfg.find_functions(self, root_node)
                if cfg.find_functions
                else []
            )
            classes = (
                cfg.find_classes(self, root_node)
                if cfg.find_classes
                else []
            )
            imports = (
                cfg.find_imports(self, root_node)
                if cfg.find_imports
                else []
            )
            # Variables before calls so calls hooks can use type inference
            variables = (
                cfg.find_variables(self, root_node)
                if cfg.find_variables
                else self._default_find_variables(root_node)
            )
            self._parsed_variables = variables
            calls = (
                cfg.find_calls(self, root_node)
                if cfg.find_calls
                else self._default_find_calls(root_node)
            )

            result: dict[str, Any] = {
                "path": str(original_path),
                "functions": functions,
                "classes": classes,
                "variables": variables,
                "imports": imports,
                "function_calls": calls,
                "is_dependency": is_dependency,
                "lang": self.language_name,
            }

            # Extra entity types (interfaces, traits, type_aliases, etc.)
            for key, finder in cfg.find_extra.items():
                result[key] = finder(self, root_node)

            return result

        except Exception as e:
            error_logger(f"Failed to parse {original_path}: {e}")
            return {"path": str(original_path), "error": str(e)}

        finally:
            if cleanup_fn:
                cleanup_fn()

    def _empty_result(self, path: str, is_dependency: bool) -> dict[str, Any]:
        result: dict[str, Any] = {
            "path": path,
            "functions": [],
            "classes": [],
            "variables": [],
            "imports": [],
            "function_calls": [],
            "is_dependency": is_dependency,
            "lang": self.language_name,
        }
        for key in self.config.find_extra:
            result[key] = []
        return result


# ---------------------------------------------------------------------------
# Simple pre-scan factory
# ---------------------------------------------------------------------------

def make_pre_scan(query_str: str) -> PreScanFn:
    """Create a pre_scan_* function from a tree-sitter query string."""

    def pre_scan(files: list[Path], parser_wrapper: Any) -> dict[str, list[str]]:
        imports_map: dict[str, list[str]] = {}
        for path in files:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    tree = parser_wrapper.parser.parse(bytes(f.read(), "utf8"))
                for capture, _ in execute_query(
                    parser_wrapper.language, query_str, tree.root_node
                ):
                    name = capture.text.decode("utf-8")
                    if name not in imports_map:
                        imports_map[name] = []
                    imports_map[name].append(str(path.resolve()))
            except Exception as e:
                warning_logger(f"Tree-sitter pre-scan failed for {path}: {e}")
        return imports_map

    return pre_scan
