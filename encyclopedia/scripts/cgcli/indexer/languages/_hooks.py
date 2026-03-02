"""Language-specific hook functions for the data-driven parser system.

Extracted from per-language parser classes into standalone functions
compatible with the GenericExtractor / LanguageConfig system.

Languages: Python, JavaScript, TypeScript, TypeScriptJSX, Go, Rust,
Ruby, Java, C, C++, C#, PHP, Swift, Kotlin, Scala, Haskell.

Every finder hook matches FinderFn:
    def hook_name(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]

Pre-scan functions match PreScanFn:
    def pre_scan_xxx(files: list[Path], parser_wrapper: Any) -> dict[str, list[str]]
"""
from __future__ import annotations

import ast
import logging
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..debug_log import debug_log, error_logger, info_logger, warning_logger
from ..tree_sitter_manager import execute_query

if TYPE_CHECKING:
    from ._base import GenericExtractor

# ---------------------------------------------------------------------------
# Notebook support (optional dependency)
# ---------------------------------------------------------------------------

try:
    import nbformat
    from nbconvert import PythonExporter

    HAS_NOTEBOOK_SUPPORT = True
    # Suppress verbose traitlets/nbconvert DEBUG logs
    logging.getLogger("traitlets").setLevel(logging.WARNING)
    logging.getLogger("nbconvert").setLevel(logging.WARNING)
except ImportError:
    HAS_NOTEBOOK_SUPPORT = False

# Suppress IPython UserWarning from nbconvert
warnings.filterwarnings("ignore", message=".*IPython is needed to transform IPython syntax.*")


# ###########################################################################
# Shared JS/TS helpers
# ###########################################################################

_GETTER_RE = re.compile(r"^\s*(?:static\s+)?get\b")
_SETTER_RE = re.compile(r"^\s*(?:static\s+)?set\b")
_STATIC_RE = re.compile(r"^\s*static\b")


def _first_line_before_body(text: str) -> str:
    """
    Best-effort header extraction: take text before the first '{'
    (covers class/object methods). Fallback to the first line.
    """
    head = text.split("{", 1)[0]
    if not head.strip():
        return text.splitlines()[0] if text.splitlines() else text
    return head


def _classify_method_kind(header: str) -> str | None:
    """
    Return 'getter' | 'setter' | 'static' | None.
    Prefer 'getter'/'setter' over 'static' when both appear.
    """
    if _GETTER_RE.search(header):
        return "getter"
    if _SETTER_RE.search(header):
        return "setter"
    if _STATIC_RE.search(header):
        return "static"
    return None


# ---------------------------------------------------------------------------
# Shared JS/TS internal helpers (used by find_functions hooks)
# ---------------------------------------------------------------------------

def _js_ts_fn_for_name(name_node: Any) -> Any | None:
    """Find the function AST node that owns *name_node*."""
    current = name_node.parent
    while current:
        if current.type in (
            "function_declaration",
            "function",
            "arrow_function",
            "method_definition",
            "function_expression",
        ):
            return current
        elif current.type in ("variable_declarator", "assignment_expression"):
            for child in current.children:
                if child.type in ("function", "arrow_function", "function_expression"):
                    return child
        current = current.parent
    return None


def _js_ts_fn_for_params(params_node: Any) -> Any | None:
    """Find the function AST node that owns *params_node*."""
    current = params_node.parent
    while current:
        if current.type in (
            "function_declaration",
            "function",
            "arrow_function",
            "method_definition",
            "function_expression",
        ):
            return current
        current = current.parent
    return None


def _js_ts_key(n: Any) -> tuple[int, int, str]:
    """Stable key for deduplicating function nodes across captures."""
    return (n.start_byte, n.end_byte, n.type)


def _js_extract_parameters(ext: GenericExtractor, params_node: Any) -> list[str]:
    """Extract parameter names from a JS formal_parameters node."""
    params: list[str] = []
    if params_node.type == "formal_parameters":
        for child in params_node.children:
            if child.type == "identifier":
                params.append(ext.get_node_text(child))
            elif child.type == "assignment_pattern":
                # Default parameter: param = defaultValue
                left_child = child.child_by_field_name("left")
                if left_child and left_child.type == "identifier":
                    params.append(ext.get_node_text(left_child))
            elif child.type == "rest_pattern":
                # Rest parameter: ...args
                argument = child.child_by_field_name("argument")
                if argument and argument.type == "identifier":
                    params.append(f"...{ext.get_node_text(argument)}")
    return params


def _ts_extract_parameters(ext: GenericExtractor, params_node: Any) -> list[str]:
    """Extract parameter names from a TS formal_parameters node."""
    params: list[str] = []
    if params_node.type == "formal_parameters":
        for child in params_node.children:
            if child.type == "identifier":
                params.append(ext.get_node_text(child))
            elif child.type == "required_parameter":
                # required_parameter -> pattern (identifier) + type_annotation
                pattern = child.child_by_field_name("pattern")
                if pattern:
                    params.append(ext.get_node_text(pattern))
                else:
                    # Fallback: first child that is an identifier or pattern
                    for sub in child.children:
                        if sub.type in ("identifier", "object_pattern", "array_pattern"):
                            params.append(ext.get_node_text(sub))
                            break
            elif child.type == "optional_parameter":
                pattern = child.child_by_field_name("pattern")
                if pattern:
                    params.append(ext.get_node_text(pattern))
            elif child.type == "assignment_pattern":
                left_child = child.child_by_field_name("left")
                if left_child and left_child.type == "identifier":
                    params.append(ext.get_node_text(left_child))
            elif child.type == "rest_pattern":
                argument = child.child_by_field_name("argument")
                if argument and argument.type == "identifier":
                    params.append(f"...{ext.get_node_text(argument)}")
    return params


def _js_get_jsdoc_comment(ext: GenericExtractor, func_node: Any) -> str | None:
    """Extract JSDoc comment preceding the function."""
    prev_sibling = func_node.prev_sibling
    while prev_sibling and prev_sibling.type in ("comment", "\n", " "):
        if prev_sibling.type == "comment":
            comment_text = ext.get_node_text(prev_sibling)
            if comment_text.startswith("/**") and comment_text.endswith("*/"):
                return comment_text.strip()
        prev_sibling = prev_sibling.prev_sibling
    return None


# ###########################################################################
# PYTHON hooks
# ###########################################################################


def find_python_functions(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find Python function definitions and lambda assignments."""
    functions = _find_python_function_defs(ext, root_node)
    functions.extend(_find_python_lambda_assignments(ext, root_node))
    return functions


def _find_python_function_defs(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Core Python function_definition finder."""
    functions: list[dict[str, Any]] = []
    query_str = ext.config.queries["functions"]
    for match in ext.run_query(query_str, root_node):
        capture_name = match[1]
        node = match[0]

        if capture_name == "name":
            func_node = node.parent
            name = ext.get_node_text(node)
            params_node = func_node.child_by_field_name("parameters")
            body_node = func_node.child_by_field_name("body")

            decorators = [
                ext.get_node_text(child)
                for child in func_node.children
                if child.type == "decorator"
            ]

            context, context_type, _ = ext.get_parent_context(func_node)
            class_context, _, _ = ext.get_parent_context(
                func_node, types=("class_definition",)
            )

            args: list[str] = []
            if params_node:
                for p in params_node.children:
                    arg_text = None
                    if p.type == "identifier":
                        # Simple parameter: def foo(x)
                        arg_text = ext.get_node_text(p)
                    elif p.type == "default_parameter":
                        # Parameter with default: def foo(x=5)
                        name_node = p.child_by_field_name("name")
                        if name_node:
                            arg_text = ext.get_node_text(name_node)
                    elif p.type == "typed_parameter":
                        # Typed parameter: def foo(x: int)
                        name_node = p.child_by_field_name("name")
                        if name_node:
                            arg_text = ext.get_node_text(name_node)
                    elif p.type == "typed_default_parameter":
                        # Typed parameter with default: def foo(x: int = 5)
                        name_node = p.child_by_field_name("name")
                        if name_node:
                            arg_text = ext.get_node_text(name_node)
                    elif p.type == "list_splat_pattern" or p.type == "dictionary_splat_pattern":
                        # *args or **kwargs
                        arg_text = ext.get_node_text(p)

                    if arg_text:
                        args.append(arg_text)

            func_data: dict[str, Any] = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "end_line": func_node.end_point[0] + 1,
                "args": args,
                "cyclomatic_complexity": ext.calculate_complexity(func_node),
                "context": context,
                "context_type": context_type,
                "class_context": class_context,
                "decorators": [d for d in decorators if d],
                "lang": ext.language_name,
                "is_dependency": False,
            }

            if ext.index_source:
                func_data["source"] = ext.get_node_text(func_node)
                func_data["docstring"] = ext.get_docstring(func_node, body_node)

            functions.append(func_data)
    return functions


def _find_python_lambda_assignments(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find lambda expressions assigned to variables (treated as functions)."""
    functions: list[dict[str, Any]] = []
    query_str = ext.config.queries.get("lambda_assignments")
    if not query_str:
        return []

    for match in ext.run_query(query_str, root_node):
        capture_name = match[1]
        node = match[0]

        if capture_name == "name":
            assignment_node = node.parent
            lambda_node = assignment_node.child_by_field_name("right")
            name = ext.get_node_text(node)
            params_node = lambda_node.child_by_field_name("parameters")

            context, context_type, _ = ext.get_parent_context(assignment_node)
            class_context, _, _ = ext.get_parent_context(
                assignment_node, types=("class_definition",)
            )

            func_data: dict[str, Any] = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "end_line": assignment_node.end_point[0] + 1,
                "args": (
                    [
                        p
                        for p in [
                            ext.get_node_text(p)
                            for p in params_node.children
                            if p.type == "identifier"
                        ]
                        if p
                    ]
                    if params_node
                    else []
                ),
                "cyclomatic_complexity": 1,
                "context": context,
                "context_type": context_type,
                "class_context": class_context,
                "decorators": [],
                "lang": ext.language_name,
                "is_dependency": False,
            }
            if ext.index_source:
                func_data["source"] = ext.get_node_text(assignment_node)
                func_data["docstring"] = None

            functions.append(func_data)
    return functions


def find_python_classes(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find Python class definitions."""
    classes: list[dict[str, Any]] = []
    query_str = ext.config.queries["classes"]
    for match in ext.run_query(query_str, root_node):
        capture_name = match[1]
        node = match[0]

        if capture_name == "name":
            class_node = node.parent
            name = ext.get_node_text(node)
            body_node = class_node.child_by_field_name("body")
            superclasses_node = class_node.child_by_field_name("superclasses")

            bases: list[str] = []
            if superclasses_node:
                bases = [
                    ext.get_node_text(child)
                    for child in superclasses_node.children
                    if child.type in ("identifier", "attribute")
                ]

            decorators = [
                ext.get_node_text(child)
                for child in class_node.children
                if child.type == "decorator"
            ]

            context, _, _ = ext.get_parent_context(class_node)

            class_data: dict[str, Any] = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "end_line": class_node.end_point[0] + 1,
                "bases": [b for b in bases if b],
                "context": context,
                "decorators": [d for d in decorators if d],
                "lang": ext.language_name,
                "is_dependency": False,
            }
            if ext.index_source:
                class_data["source"] = ext.get_node_text(class_node)
                class_data["docstring"] = ext.get_docstring(class_node, body_node)

            classes.append(class_data)
    return classes


def find_python_imports(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find Python import and from-import statements."""
    imports: list[dict[str, Any]] = []
    seen_modules: set[str] = set()
    query_str = ext.config.queries["imports"]
    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name in ("import", "from_import_stmt"):
            # For 'import_statement'
            if capture_name == "import":
                node_text = ext.get_node_text(node)
                alias = None
                if " as " in node_text:
                    parts = node_text.split(" as ")
                    full_name = parts[0].strip()
                    alias = parts[1].strip()
                else:
                    full_name = node_text.strip()

                if full_name in seen_modules:
                    continue
                seen_modules.add(full_name)

                import_data = {
                    "name": full_name,
                    "full_import_name": full_name,
                    "line_number": node.start_point[0] + 1,
                    "alias": alias,
                    "context": ext.get_parent_context(node)[:2],
                    "lang": ext.language_name,
                    "is_dependency": False,
                }
                imports.append(import_data)
            # For 'import_from_statement'
            elif capture_name == "from_import_stmt":
                module_name_node = node.child_by_field_name("module_name")
                if not module_name_node:
                    continue

                module_name = ext.get_node_text(module_name_node)

                # Handle 'from ... import ...'
                import_list_node = node.child_by_field_name("name")
                if import_list_node:
                    for child in import_list_node.children:
                        imported_name = None
                        alias = None
                        if child.type == "aliased_import":
                            name_node = child.child_by_field_name("name")
                            alias_node = child.child_by_field_name("alias")
                            if name_node:
                                imported_name = ext.get_node_text(name_node)
                            if alias_node:
                                alias = ext.get_node_text(alias_node)
                        elif child.type == "dotted_name" or child.type == "identifier":
                            imported_name = ext.get_node_text(child)

                        if imported_name:
                            full_import_name = f"{module_name}.{imported_name}"
                            if full_import_name in seen_modules:
                                continue
                            seen_modules.add(full_import_name)
                            imports.append(
                                {
                                    "name": imported_name,
                                    "full_import_name": full_import_name,
                                    "line_number": child.start_point[0] + 1,
                                    "alias": alias,
                                    "context": ext.get_parent_context(child)[:2],
                                    "lang": ext.language_name,
                                    "is_dependency": False,
                                }
                            )

    return imports


def find_python_calls(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find Python function/method calls including dict-method indirect references."""
    calls: list[dict[str, Any]] = []

    # First, find all direct function calls
    query_str = ext.config.queries["calls"]
    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "name":
            call_node = (
                node.parent if node.parent.type == "call" else node.parent.parent
            )
            full_call_node = call_node.child_by_field_name("function")

            args: list[str] = []
            arguments_node = call_node.child_by_field_name("arguments")
            if arguments_node:
                for arg in arguments_node.children:
                    arg_text = ext.get_node_text(arg)
                    if arg_text and arg_text not in ("(", ")", ","):
                        args.append(arg_text)

            call_data = {
                "name": ext.get_node_text(node),
                "full_name": ext.get_node_text(full_call_node),
                "line_number": node.start_point[0] + 1,
                "args": args,
                "inferred_obj_type": None,
                "context": ext.get_parent_context(node),
                "class_context": ext.get_parent_context(
                    node, types=("class_definition",)
                )[:2],
                "lang": ext.language_name,
                "is_dependency": False,
            }
            calls.append(call_data)

    # Second, find dictionary-based method references (indirect calls)
    dict_method_calls = _find_python_dict_method_references(ext, root_node)
    calls.extend(dict_method_calls)

    return calls


def _find_python_dict_method_references(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """
    Detects indirect function calls through dictionary mappings.

    Example pattern:
        tool_map = {
            "add_code": self.add_code_to_graph_tool,
            "find_code": self.find_code_tool,
        }
        handler = tool_map.get(tool_name)
        if handler:
            handler(**args)

    This creates CALLS relationships from the context function to all
    methods referenced in the dictionary.
    """
    calls: list[dict[str, Any]] = []
    query_str = ext.config.queries.get("dict_method_refs")
    if not query_str:
        return calls

    # Track dictionaries that contain method references
    dict_assignments: dict[str, Any] = {}  # dict_var_name -> list of method references

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "method_ref":
            # Found a method reference in a dictionary value
            # Navigate up to find the assignment
            dict_node = node.parent  # pair node
            while dict_node and dict_node.type != "dictionary":
                dict_node = dict_node.parent

            if dict_node:
                # Find the assignment node
                assignment_node = dict_node.parent
                if assignment_node and assignment_node.type == "assignment":
                    # Get the variable name being assigned
                    left_node = assignment_node.child_by_field_name("left")
                    if left_node:
                        var_name = ext.get_node_text(left_node)
                        method_ref = ext.get_node_text(node)

                        # Extract just the method name (remove 'self.')
                        method_name = (
                            method_ref.split(".")[-1] if "." in method_ref else method_ref
                        )

                        if var_name not in dict_assignments:
                            dict_assignments[var_name] = {
                                "methods": [],
                                "context": ext.get_parent_context(assignment_node),
                                "line_number": assignment_node.start_point[0] + 1,
                            }

                        dict_assignments[var_name]["methods"].append(
                            {
                                "name": method_name,
                                "full_name": method_ref,
                                "line_number": node.start_point[0] + 1,
                            }
                        )

    # Now create call relationships for each method in the dictionaries
    # The context is the function where the dictionary is defined
    for dict_var, data in dict_assignments.items():
        context, context_type, context_line = data["context"]
        class_context, _, _ = (None, None, None)

        for method_info in data["methods"]:
            call_data = {
                "name": method_info["name"],
                "full_name": method_info["full_name"],
                "line_number": method_info["line_number"],
                "args": [],  # We don't know the args at this point
                "inferred_obj_type": None,
                "context": (context, context_type, context_line),
                "class_context": (class_context, None),
                "lang": ext.language_name,
                "is_dependency": False,
                "is_indirect_call": True,  # Mark as indirect for debugging
            }
            calls.append(call_data)

    return calls


def find_python_variables(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find Python variable assignments (excluding lambdas)."""
    variables: list[dict[str, Any]] = []
    query_str = ext.config.queries["variables"]
    for match in ext.run_query(query_str, root_node):
        capture_name = match[1]
        node = match[0]

        if capture_name == "name":
            assignment_node = node.parent

            # Skip lambda assignments, they are handled by _find_python_lambda_assignments
            right_node = assignment_node.child_by_field_name("right")
            if right_node and right_node.type == "lambda":
                continue

            name = ext.get_node_text(node)
            value = ext.get_node_text(right_node) if right_node else None

            type_node = assignment_node.child_by_field_name("type")
            type_text = ext.get_node_text(type_node) if type_node else None

            context, _, _ = ext.get_parent_context(node)
            class_context, _, _ = ext.get_parent_context(
                node, types=("class_definition",)
            )

            variable_data = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "value": value,
                "type": type_text,
                "context": context,
                "class_context": class_context,
                "lang": ext.language_name,
                "is_dependency": False,
            }
            variables.append(variable_data)
    return variables


# ---------------------------------------------------------------------------
# Python pre-parse hook (notebook conversion)
# ---------------------------------------------------------------------------


def pre_parse_python_notebook(
    path: Path, parser: Any, **kwargs: Any
) -> tuple[str, Path, Callable[[], None] | None]:
    """Convert .ipynb to a temp .py file, or read .py directly."""
    is_notebook = kwargs.get("is_notebook", False)
    temp_py_file: Path | None = None

    if is_notebook:
        if not HAS_NOTEBOOK_SUPPORT:
            # Return empty source so the caller can produce an error result
            raise RuntimeError(
                "nbformat/nbconvert not installed — cannot parse notebooks"
            )
        info_logger(f"Converting notebook {path} to temporary Python file.")
        with open(path, "r", encoding="utf-8") as f:
            notebook_node = nbformat.read(f, as_version=4)

        exporter = PythonExporter()
        python_code, _ = exporter.from_notebook_node(notebook_node)

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".py", encoding="utf-8"
        ) as tf:
            tf.write(python_code)
            temp_py_file = Path(tf.name)

        with open(temp_py_file, "r", encoding="utf-8") as f:
            source_code = f.read()

        def _cleanup() -> None:
            if temp_py_file and temp_py_file.exists():
                os.remove(temp_py_file)
                info_logger(f"Removed temporary file: {temp_py_file}")

        return source_code, temp_py_file, _cleanup

    # Regular .py file — no cleanup needed
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        source_code = f.read()
    return source_code, path, None


# ---------------------------------------------------------------------------
# Python pre-scan
# ---------------------------------------------------------------------------


def pre_scan_python(
    files: list[Path], parser_wrapper: Any
) -> dict[str, list[str]]:
    """Scans Python files to create a map of class/function names to their file paths."""
    imports_map: dict[str, list[str]] = {}
    query_str = """
        (class_definition name: (identifier) @name)
        (function_definition name: (identifier) @name)
    """

    for path in files:
        temp_py_file = None
        try:
            source_to_parse = ""
            if path.suffix == ".ipynb":
                if not HAS_NOTEBOOK_SUPPORT:
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    notebook_node = nbformat.read(f, as_version=4)
                exporter = PythonExporter()
                python_code, _ = exporter.from_notebook_node(notebook_node)
                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".py", encoding="utf-8"
                ) as tf:
                    tf.write(python_code)
                    temp_py_file = Path(tf.name)
                with open(temp_py_file, "r", encoding="utf-8") as f:
                    source_to_parse = f.read()
            else:
                with open(path, "r", encoding="utf-8") as f:
                    source_to_parse = f.read()

            tree = parser_wrapper.parser.parse(bytes(source_to_parse, "utf8"))

            for capture, _ in execute_query(
                parser_wrapper.language, query_str, tree.root_node
            ):
                name = capture.text.decode("utf-8")
                if name not in imports_map:
                    imports_map[name] = []
                imports_map[name].append(str(path.resolve()))
        except Exception as e:
            warning_logger(f"Tree-sitter pre-scan failed for {path}: {e}")
        finally:
            if temp_py_file and temp_py_file.exists():
                os.remove(temp_py_file)
    return imports_map


# ###########################################################################
# JAVASCRIPT hooks
# ###########################################################################


def find_js_functions(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find JavaScript function declarations, expressions, arrows, and methods."""
    functions: list[dict[str, Any]] = []
    query_str = ext.config.queries["functions"]

    # Collect captures grouped by function node
    captures_by_function: dict[tuple[int, int, str], dict[str, Any]] = {}

    def _bucket_for(node: Any) -> dict[str, Any]:
        fid = _js_ts_key(node)
        return captures_by_function.setdefault(
            fid, {"node": node, "name": None, "params": None, "single_param": None}
        )

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "function_node":
            _bucket_for(node)
        elif capture_name == "name":
            fn = _js_ts_fn_for_name(node)
            if fn:
                b = _bucket_for(fn)
                b["name"] = ext.get_node_text(node)
        elif capture_name == "params":
            fn = _js_ts_fn_for_params(node)
            if fn:
                b = _bucket_for(fn)
                b["params"] = node
        elif capture_name == "single_param":
            fn = _js_ts_fn_for_params(node)
            if fn:
                b = _bucket_for(fn)
                b["single_param"] = node

    # Build Function entries
    for _, data in captures_by_function.items():
        func_node = data["node"]

        # Backfill name for method_definition if query didn't capture it
        name = data.get("name")
        if not name and func_node.type == "method_definition":
            nm = func_node.child_by_field_name("name")
            if nm:
                name = ext.get_node_text(nm)
        if not name:
            continue  # skip nameless functions

        # Parameters
        args: list[str] = []
        if data.get("params"):
            args = _js_extract_parameters(ext, data["params"])
        elif data.get("single_param"):
            args = [ext.get_node_text(data["single_param"])]

        # Context & docstring
        context, context_type, _ = ext.get_parent_context(func_node)
        class_context = context if context_type == "class_declaration" else None
        docstring = _js_get_jsdoc_comment(ext, func_node)

        # Classify getter/setter/static (methods only)
        js_kind = None
        if func_node.type == "method_definition":
            header = _first_line_before_body(ext.get_node_text(func_node))
            js_kind = _classify_method_kind(header)

        func_data: dict[str, Any] = {
            "name": name,
            "line_number": func_node.start_point[0] + 1,
            "end_line": func_node.end_point[0] + 1,
            "args": args,
            "lang": ext.language_name,
            "is_dependency": False,
        }

        if ext.index_source:
            func_data["source"] = ext.get_node_text(func_node)
            func_data["docstring"] = docstring
        if js_kind is not None:
            func_data["type"] = js_kind

        functions.append(func_data)

    return functions


def find_js_classes(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find JavaScript class declarations."""
    classes: list[dict[str, Any]] = []
    query_str = ext.config.queries["classes"]
    for class_node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "class":
            name_node = class_node.child_by_field_name("name")
            if not name_node:
                continue
            name = ext.get_node_text(name_node)

            bases: list[str] = []
            heritage_node = next(
                (child for child in class_node.children if child.type == "class_heritage"),
                None,
            )
            if heritage_node:
                if heritage_node.named_child_count > 0:
                    base_expr_node = heritage_node.named_child(0)
                    bases.append(ext.get_node_text(base_expr_node))
                elif heritage_node.child_count > 0:
                    # Fallback for anonymous nodes
                    base_expr_node = heritage_node.child(heritage_node.child_count - 1)
                    bases.append(ext.get_node_text(base_expr_node))

            class_data: dict[str, Any] = {
                "name": name,
                "line_number": class_node.start_point[0] + 1,
                "end_line": class_node.end_point[0] + 1,
                "bases": bases,
                "context": None,
                "decorators": [],
                "lang": ext.language_name,
                "is_dependency": False,
            }

            if ext.index_source:
                class_data["source"] = ext.get_node_text(class_node)
                class_data["docstring"] = ext.get_docstring(class_node)

            classes.append(class_data)
    return classes


def find_js_imports(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find JavaScript import statements and require() calls."""
    imports: list[dict[str, Any]] = []
    query_str = ext.config.queries["imports"]
    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name != "import":
            continue

        line_number = node.start_point[0] + 1

        if node.type == "import_statement":
            source = ext.get_node_text(node.child_by_field_name("source")).strip("'\"")

            # Look for different import structures
            import_clause = node.child_by_field_name("import")
            if not import_clause:
                imports.append(
                    {
                        "name": source,
                        "source": source,
                        "alias": None,
                        "line_number": line_number,
                        "lang": ext.language_name,
                    }
                )
                continue

            # Default import: import defaultExport from '...'
            if import_clause.type == "identifier":
                alias = ext.get_node_text(import_clause)
                imports.append(
                    {
                        "name": "default",
                        "source": source,
                        "alias": alias,
                        "line_number": line_number,
                        "lang": ext.language_name,
                    }
                )

            # Namespace import: import * as name from '...'
            elif import_clause.type == "namespace_import":
                alias_node = import_clause.child_by_field_name("alias")
                if alias_node:
                    alias = ext.get_node_text(alias_node)
                    imports.append(
                        {
                            "name": "*",
                            "source": source,
                            "alias": alias,
                            "line_number": line_number,
                            "lang": ext.language_name,
                        }
                    )

            # Named imports: import { name, name as alias } from '...'
            elif import_clause.type == "named_imports":
                for specifier in import_clause.children:
                    if specifier.type == "import_specifier":
                        name_node = specifier.child_by_field_name("name")
                        alias_node = specifier.child_by_field_name("alias")
                        original_name = ext.get_node_text(name_node)
                        alias = ext.get_node_text(alias_node) if alias_node else None
                        imports.append(
                            {
                                "name": original_name,
                                "source": source,
                                "alias": alias,
                                "line_number": line_number,
                                "lang": ext.language_name,
                            }
                        )

        elif node.type == "call_expression":  # require('...')
            args = node.child_by_field_name("arguments")
            if not args or args.named_child_count == 0:
                continue
            source_node = args.named_child(0)
            if not source_node or source_node.type != "string":
                continue
            source = ext.get_node_text(source_node).strip("'\"")

            alias = None
            if node.parent.type == "variable_declarator":
                alias_node = node.parent.child_by_field_name("name")
                if alias_node:
                    alias = ext.get_node_text(alias_node)
            imports.append(
                {
                    "name": source,
                    "source": source,
                    "alias": alias,
                    "line_number": line_number,
                    "lang": ext.language_name,
                }
            )

    return imports


def find_js_calls(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find JavaScript function/method calls."""
    calls: list[dict[str, Any]] = []
    query_str = ext.config.queries["calls"]
    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "name":
            # Traverse up to find the call_expression
            call_node = node.parent
            while (
                call_node
                and call_node.type != "call_expression"
                and call_node.type != "program"
            ):
                call_node = call_node.parent

            name = ext.get_node_text(node)

            # Improved args extraction
            args: list[str] = []
            arguments_node = None
            if call_node and call_node.type in ("call_expression", "new_expression"):
                arguments_node = call_node.child_by_field_name("arguments")

            if arguments_node:
                for arg in arguments_node.children:
                    if arg.type not in ("(", ")", ","):
                        args.append(ext.get_node_text(arg))

            call_data = {
                "name": name,
                "full_name": ext.get_node_text(call_node),
                "line_number": node.start_point[0] + 1,
                "args": args,
                "inferred_obj_type": None,
                "context": ext.get_parent_context(node),
                "class_context": ext.get_parent_context(
                    node, types=("class_declaration",)
                )[:2],
                "lang": ext.language_name,
                "is_dependency": False,
            }
            calls.append(call_data)
    return calls


def find_js_variables(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find JavaScript variable declarations (excluding function assignments)."""
    variables: list[dict[str, Any]] = []
    query_str = ext.config.queries["variables"]
    for match in ext.run_query(query_str, root_node):
        capture_name = match[1]
        node = match[0]

        if capture_name == "name":
            var_node = node.parent
            name = ext.get_node_text(node)
            value = None
            type_text = None

            # Detect if variable assigned to a function
            value_node = var_node.child_by_field_name("value") if var_node else None

            if value_node:
                value_type = value_node.type

                # --- Skip variables that are assigned a function ---
                if value_type in ("function_expression", "arrow_function"):
                    continue

                # Some grammars might have async_arrow_function or similar
                if "function" in value_type or "arrow" in value_type:
                    continue

                # --- Handle various assignment types ---
                if value_type == "call_expression":
                    func_node = value_node.child_by_field_name("function")
                    value = ext.get_node_text(func_node) if func_node else name
                else:
                    value = ext.get_node_text(value_node)

            context, context_type, context_line = ext.get_parent_context(node)
            class_context = context if context_type == "class_declaration" else None

            variable_data = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "value": value,
                "type": type_text,
                "context": context,
                "class_context": class_context,
                "lang": ext.language_name,
                "is_dependency": False,
            }
            variables.append(variable_data)
    return variables


# ---------------------------------------------------------------------------
# JavaScript pre-scan
# ---------------------------------------------------------------------------


def pre_scan_javascript(
    files: list[Path], parser_wrapper: Any
) -> dict[str, list[str]]:
    """Scans JavaScript files to create a map of class/function names to their file paths."""
    imports_map: dict[str, list[str]] = {}
    query_str = """
        (class_declaration name: (identifier) @name)
        (function_declaration name: (identifier) @name)
        (variable_declarator name: (identifier) @name value: (function_expression))
        (variable_declarator name: (identifier) @name value: (arrow_function))
        (method_definition name: (property_identifier) @name)
        (assignment_expression
            left: (member_expression
                property: (property_identifier) @name
            )
            right: (function_expression)
        )
        (assignment_expression
            left: (member_expression
                property: (property_identifier) @name
            )
            right: (arrow_function)
        )
    """

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
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


# ###########################################################################
# TYPESCRIPT hooks
# ###########################################################################


def find_ts_functions(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find TypeScript function declarations, expressions, arrows, and methods."""
    functions: list[dict[str, Any]] = []
    query_str = ext.config.queries["functions"]

    captures_by_function: dict[tuple[int, int, str], dict[str, Any]] = {}

    def _bucket_for(node: Any) -> dict[str, Any]:
        fid = _js_ts_key(node)
        return captures_by_function.setdefault(
            fid, {"node": node, "name": None, "params": None, "single_param": None}
        )

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "function_node":
            _bucket_for(node)
        elif capture_name == "name":
            fn = _js_ts_fn_for_name(node)
            if fn:
                b = _bucket_for(fn)
                b["name"] = ext.get_node_text(node)
        elif capture_name == "params":
            fn = _js_ts_fn_for_params(node)
            if fn:
                b = _bucket_for(fn)
                b["params"] = node
        elif capture_name == "single_param":
            fn = _js_ts_fn_for_params(node)
            if fn:
                b = _bucket_for(fn)
                b["single_param"] = node

    for _, data in captures_by_function.items():
        func_node = data["node"]
        name = data.get("name")
        if not name and func_node.type == "method_definition":
            nm = func_node.child_by_field_name("name")
            if nm:
                name = ext.get_node_text(nm)
        if not name:
            continue

        args: list[str] = []
        if data.get("params"):
            args = _ts_extract_parameters(ext, data["params"])
        elif data.get("single_param"):
            args = [ext.get_node_text(data["single_param"])]

        context, context_type, _ = ext.get_parent_context(func_node)
        class_context = context if context_type == "class_declaration" else None
        docstring = None

        func_data: dict[str, Any] = {
            "name": name,
            "line_number": func_node.start_point[0] + 1,
            "end_line": func_node.end_point[0] + 1,
            "args": args,
            "args": args,
            "cyclomatic_complexity": ext.calculate_complexity(func_node),
            "context": context,
            "context_type": context_type,
            "class_context": class_context,
            "decorators": [],
            "lang": ext.language_name,
            "is_dependency": False,
        }

        if ext.index_source:
            func_data["source"] = ext.get_node_text(func_node)
            func_data["docstring"] = docstring
        functions.append(func_data)
    return functions


def find_ts_classes(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find TypeScript class and abstract class declarations."""
    classes: list[dict[str, Any]] = []
    query_str = ext.config.queries["classes"]
    for class_node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "class":
            name_node = class_node.child_by_field_name("name")
            if not name_node:
                continue
            name = ext.get_node_text(name_node)
            bases: list[str] = []
            heritage_node = next(
                (child for child in class_node.children if child.type == "class_heritage"),
                None,
            )
            if heritage_node:
                for child in heritage_node.children:
                    if child.type == "extends_clause":
                        # extends_clause -> extends identifier
                        for sub in child.children:
                            if sub.type in (
                                "identifier",
                                "type_identifier",
                                "member_expression",
                            ):
                                bases.append(ext.get_node_text(sub))
                    elif child.type == "implements_clause":
                        # implements_clause -> implements identifier, identifier...
                        for sub in child.children:
                            if sub.type in (
                                "identifier",
                                "type_identifier",
                                "member_expression",
                            ):
                                bases.append(ext.get_node_text(sub))
            class_data: dict[str, Any] = {
                "name": name,
                "line_number": class_node.start_point[0] + 1,
                "end_line": class_node.end_point[0] + 1,
                "bases": bases,
                "bases": bases,
                "context": None,
                "decorators": [],
                "lang": ext.language_name,
                "is_dependency": False,
            }
            if ext.index_source:
                class_data["source"] = ext.get_node_text(class_node)
                class_data["docstring"] = ext.get_docstring(class_node)
            classes.append(class_data)
    return classes


def find_ts_interfaces(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find TypeScript interface declarations."""
    interfaces: list[dict[str, Any]] = []
    query_str = ext.config.queries["interfaces"]
    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "interface_node":
            name_node = node.child_by_field_name("name")
            if not name_node:
                continue

            name = ext.get_node_text(name_node)
            interface_data: dict[str, Any] = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "end_line": node.end_point[0] + 1,
            }
            if ext.index_source:
                interface_data["source"] = ext.get_node_text(node)
            interfaces.append(interface_data)
    return interfaces


def find_ts_type_aliases(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find TypeScript type alias declarations."""
    type_aliases: list[dict[str, Any]] = []
    query_str = ext.config.queries["type_aliases"]
    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "type_alias_node":
            name_node = node.child_by_field_name("name")
            if not name_node:
                continue

            name = ext.get_node_text(name_node)
            type_alias_data: dict[str, Any] = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "end_line": node.end_point[0] + 1,
            }
            if ext.index_source:
                type_alias_data["source"] = ext.get_node_text(node)
            type_aliases.append(type_alias_data)
    return type_aliases


def find_ts_imports(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find TypeScript import statements and require() calls."""
    imports: list[dict[str, Any]] = []
    query_str = ext.config.queries["imports"]
    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name != "import":
            continue
        line_number = node.start_point[0] + 1
        if node.type == "import_statement":
            source = ext.get_node_text(node.child_by_field_name("source")).strip("'\"")
            import_clause = node.child_by_field_name("import")
            if not import_clause:
                imports.append(
                    {
                        "name": source,
                        "source": source,
                        "alias": None,
                        "line_number": line_number,
                        "lang": ext.language_name,
                    }
                )
                continue
            if import_clause.type == "identifier":
                alias = ext.get_node_text(import_clause)
                imports.append(
                    {
                        "name": "default",
                        "source": source,
                        "alias": alias,
                        "line_number": line_number,
                        "lang": ext.language_name,
                    }
                )
            elif import_clause.type == "namespace_import":
                alias_node = import_clause.child_by_field_name("alias")
                if alias_node:
                    alias = ext.get_node_text(alias_node)
                    imports.append(
                        {
                            "name": "*",
                            "source": source,
                            "alias": alias,
                            "line_number": line_number,
                            "lang": ext.language_name,
                        }
                    )
            elif import_clause.type == "named_imports":
                for specifier in import_clause.children:
                    if specifier.type == "import_specifier":
                        name_node = specifier.child_by_field_name("name")
                        alias_node = specifier.child_by_field_name("alias")
                        original_name = ext.get_node_text(name_node)
                        alias = ext.get_node_text(alias_node) if alias_node else None
                        imports.append(
                            {
                                "name": original_name,
                                "source": source,
                                "alias": alias,
                                "line_number": line_number,
                                "lang": ext.language_name,
                            }
                        )
        elif node.type == "call_expression":
            args = node.child_by_field_name("arguments")
            if not args or args.named_child_count == 0:
                continue
            source_node = args.named_child(0)
            if not source_node or source_node.type != "string":
                continue
            source = ext.get_node_text(source_node).strip("'\"")
            alias = None
            if node.parent.type == "variable_declarator":
                alias_node = node.parent.child_by_field_name("name")
                if alias_node:
                    alias = ext.get_node_text(alias_node)
            imports.append(
                {
                    "name": source,
                    "source": source,
                    "alias": alias,
                    "line_number": line_number,
                    "lang": ext.language_name,
                }
            )
    return imports


def find_ts_calls(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find TypeScript function/method calls."""
    calls: list[dict[str, Any]] = []
    query_str = ext.config.queries["calls"]
    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "name":
            # Traverse up to find the call/new expression
            call_node = node.parent
            while (
                call_node
                and call_node.type not in ("call_expression", "new_expression")
                and call_node.type != "program"
            ):
                call_node = call_node.parent

            name = ext.get_node_text(node)

            # Improved args extraction
            args: list[str] = []
            arguments_node = None
            if call_node and call_node.type in ("call_expression", "new_expression"):
                arguments_node = call_node.child_by_field_name("arguments")

            if arguments_node:
                for arg in arguments_node.children:
                    if arg.type not in ("(", ")", ","):
                        args.append(ext.get_node_text(arg))

            call_data = {
                "name": name,
                "full_name": ext.get_node_text(call_node) if call_node else name,
                "line_number": node.start_point[0] + 1,
                "args": args,
                "inferred_obj_type": None,
                "context": ext.get_parent_context(node),
                "class_context": ext.get_parent_context(
                    node, types=("class_declaration", "abstract_class_declaration")
                ),
                "lang": ext.language_name,
                "is_dependency": False,
            }
            calls.append(call_data)
    return calls


def find_ts_variables(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find TypeScript variable declarations (excluding function assignments)."""
    variables: list[dict[str, Any]] = []
    query_str = ext.config.queries["variables"]
    for match in ext.run_query(query_str, root_node):
        capture_name = match[1]
        node = match[0]
        if capture_name == "name":
            var_node = node.parent
            name = ext.get_node_text(node)
            value = None
            type_text = None

            # Detect if variable assigned to a function
            value_node = var_node.child_by_field_name("value") if var_node else None

            if value_node:
                value_type = value_node.type

                # --- Skip variables that are assigned a function ---
                if value_type in ("function_expression", "arrow_function"):
                    continue

                if "function" in value_type or "arrow" in value_type:
                    continue

                # --- Handle various assignment types ---
                if value_type == "call_expression":
                    func_node = value_node.child_by_field_name("function")
                    value = ext.get_node_text(func_node) if func_node else name
                else:
                    value = ext.get_node_text(value_node)

            context, context_type, context_line = ext.get_parent_context(node)
            class_context = context if context_type == "class_declaration" else None

            variable_data = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "value": value,
                "type": type_text,
                "context": context,
                "class_context": class_context,
                "lang": ext.language_name,
                "is_dependency": False,
            }
            variables.append(variable_data)
    return variables


# ---------------------------------------------------------------------------
# TypeScript pre-scan
# ---------------------------------------------------------------------------


def pre_scan_typescript(
    files: list[Path], parser_wrapper: Any
) -> dict[str, list[str]]:
    """Scans TypeScript files to create a map of class/function names to their file paths."""
    imports_map: dict[str, list[str]] = {}

    # Simplified queries that capture the parent nodes, then extract names manually
    query_strings = [
        "(class_declaration) @class",
        "(function_declaration) @function",
        "(variable_declarator) @var_decl",
        "(method_definition) @method",
        "(interface_declaration) @interface",
        "(type_alias_declaration) @type_alias",
    ]

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                source_code = f.read()
                tree = parser_wrapper.parser.parse(bytes(source_code, "utf8"))

            # Run each query separately
            for query_str in query_strings:
                try:
                    for node, capture_name in execute_query(
                        parser_wrapper.language, query_str, tree.root_node
                    ):
                        name = None

                        # Extract name based on node type
                        if capture_name == "class":
                            name_node = node.child_by_field_name("name")
                            if name_node:
                                name = name_node.text.decode("utf-8")

                        elif capture_name == "function":
                            name_node = node.child_by_field_name("name")
                            if name_node:
                                name = name_node.text.decode("utf-8")

                        elif capture_name == "var_decl":
                            # Check if it's a function or arrow function
                            name_node = node.child_by_field_name("name")
                            value_node = node.child_by_field_name("value")
                            if name_node and value_node:
                                if value_node.type in ("function", "arrow_function"):
                                    name = name_node.text.decode("utf-8")

                        elif capture_name == "method":
                            name_node = node.child_by_field_name("name")
                            if name_node:
                                name = name_node.text.decode("utf-8")

                        elif capture_name == "interface":
                            name_node = node.child_by_field_name("name")
                            if name_node:
                                name = name_node.text.decode("utf-8")

                        elif capture_name == "type_alias":
                            name_node = node.child_by_field_name("name")
                            if name_node:
                                name = name_node.text.decode("utf-8")

                        # Add to imports map if we found a name
                        if name:
                            if name not in imports_map:
                                imports_map[name] = []
                            file_path_str = str(path.resolve())
                            if file_path_str not in imports_map[name]:
                                imports_map[name].append(file_path_str)

                except Exception as query_error:
                    warning_logger(
                        f"Query failed for pattern '{query_str}': {query_error}"
                    )

        except Exception as e:
            warning_logger(f"Tree-sitter pre-scan failed for {path}: {e}")

    return imports_map


# ###########################################################################
# TYPESCRIPTJSX hooks
# ###########################################################################

# NOTE: TypeScriptJSX reuses ALL TypeScript finder hooks (find_ts_functions,
# find_ts_classes, find_ts_interfaces, find_ts_type_aliases, find_ts_imports,
# find_ts_calls, find_ts_variables) plus the additional hook below.


def find_tsx_react_components(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """
    Find React components in .tsx files (function and class components).
    """
    components: list[dict[str, Any]] = []
    # Function components: exported arrow/function assigned to const, returning JSX
    # Class components: class extending React.Component or React.PureComponent
    # This is a simplified query, can be extended for more cases
    query_strings = [
        "(class_declaration name: (type_identifier) @name)",
        "(variable_declarator name: (identifier) @name value: (arrow_function) @fn)",
        "(variable_declarator name: (identifier) @name value: (function_expression) @fn)",
        "(function_declaration name: (identifier) @name)",
    ]
    for query_str in query_strings:
        for node, capture_name in ext.run_query(query_str, root_node):
            if capture_name == "name":
                name = node.text.decode("utf-8")
                line_number = node.start_point[0] + 1
                component_data: dict[str, Any] = {
                    "name": name,
                    "line_number": line_number,
                    "type": "component",
                    "lang": ext.language_name,
                }

                if ext.index_source:
                    # The query captures 'name', so node is the identifier.
                    # We need the parent node.
                    # (class_declaration ... name: (identifier) @name) -> parent is class_declaration
                    # (variable_declarator name: (identifier) @name ...) -> parent is variable_declarator
                    # (function_declaration name: (identifier) @name) -> parent is function_declaration
                    parent = node.parent
                    component_data["source"] = parent.text.decode("utf-8")

                components.append(component_data)
    return components


# ---------------------------------------------------------------------------
# TypeScriptJSX pre-scan
# ---------------------------------------------------------------------------


def pre_scan_typescriptjsx(
    files: list[Path], parser_wrapper: Any
) -> dict[str, list[str]]:
    """
    Scans TypeScript JSX (.tsx) files to create a map of class/function names to their file paths.
    Reuses the logic from TypeScript parser, but can be extended for JSX-specific extraction.
    """
    imports_map: dict[str, list[str]] = {}
    # Use the same queries as TypeScript
    query_strings = [
        "(class_declaration) @class",
        "(function_declaration) @function",
        "(variable_declarator) @var_decl",
        "(method_definition) @method",
        "(interface_declaration) @interface",
        "(type_alias_declaration) @type_alias",
    ]
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                source_code = f.read()
                tree = parser_wrapper.parser.parse(bytes(source_code, "utf8"))
            for query_str in query_strings:
                try:
                    for node, capture_name in execute_query(
                        parser_wrapper.language, query_str, tree.root_node
                    ):
                        name = None
                        if capture_name == "class":
                            name_node = node.child_by_field_name("name")
                            if name_node:
                                name = name_node.text.decode("utf-8")
                        elif capture_name == "function":
                            name_node = node.child_by_field_name("name")
                            if name_node:
                                name = name_node.text.decode("utf-8")
                        elif capture_name == "var_decl":
                            name_node = node.child_by_field_name("name")
                            value_node = node.child_by_field_name("value")
                            if name_node and value_node:
                                if value_node.type in ("function", "arrow_function"):
                                    name = name_node.text.decode("utf-8")
                        elif capture_name == "method":
                            name_node = node.child_by_field_name("name")
                            if name_node:
                                name = name_node.text.decode("utf-8")
                        elif capture_name == "interface":
                            name_node = node.child_by_field_name("name")
                            if name_node:
                                name = name_node.text.decode("utf-8")
                        elif capture_name == "type_alias":
                            name_node = node.child_by_field_name("name")
                            if name_node:
                                name = name_node.text.decode("utf-8")
                        if name:
                            if name not in imports_map:
                                imports_map[name] = []
                            file_path_str = str(path.resolve())
                            if file_path_str not in imports_map[name]:
                                imports_map[name].append(file_path_str)
                except Exception as query_error:
                    warning_logger(
                        f"Query failed for pattern '{query_str}': {query_error}"
                    )
        except Exception as e:
            warning_logger(f"Tree-sitter pre-scan failed for {path}: {e}")
    return imports_map


# ===================================================================
#  GO HOOKS
# ===================================================================


# -- internal helpers ------------------------------------------------


def _go_find_function_node_for_name(name_node: Any) -> Any | None:
    """Walk up the AST from a name node to find the enclosing function/method declaration."""
    current = name_node.parent
    while current:
        if current.type in ("function_declaration", "method_declaration"):
            return current
        current = current.parent
    return None


def _go_find_function_node_for_params(params_node: Any) -> Any | None:
    """Walk up the AST from a params node to find the enclosing function/method declaration."""
    current = params_node.parent
    while current:
        if current.type in ("function_declaration", "method_declaration"):
            return current
        current = current.parent
    return None


def _go_extract_parameters(ext: GenericExtractor, params_node: Any) -> list[str]:
    """Extract parameter names from a Go parameter_list node."""
    params: list[str] = []
    if params_node.type == "parameter_list":
        for child in params_node.children:
            if child.type == "parameter_declaration":
                # Handle multiple names for same type: func(x, y int)
                # We iterate children and find all identifiers that are not the type node.
                type_node = child.child_by_field_name("type")
                for grandchild in child.children:
                    if grandchild.type == "identifier":
                        if grandchild.id != (type_node.id if type_node else None):
                            params.append(ext.get_node_text(grandchild))
            elif child.type == "variadic_parameter_declaration":
                name_node = child.child_by_field_name("name")
                if name_node:
                    params.append(f"...{ext.get_node_text(name_node)}")
    return params


def _go_extract_receiver(ext: GenericExtractor, receiver_node: Any) -> str | None:
    """Extract the receiver type from a Go method receiver parameter_list."""
    if receiver_node.type == "parameter_list" and receiver_node.named_child_count > 0:
        param = receiver_node.named_child(0)
        type_node = param.child_by_field_name("type")
        if type_node:
            type_text = ext.get_node_text(type_node)
            return type_text.strip("*")
    return None


def _go_find_type_declaration_for_name(name_node: Any) -> Any | None:
    """Walk up the AST from a name node to find the enclosing type_declaration."""
    current = name_node.parent
    while current:
        if current.type == "type_declaration":
            return current
        current = current.parent
    return None


def _go_get_parent_context(
    ext: GenericExtractor, node: Any
) -> tuple[str | None, str | None, int | None]:
    """Go-specific parent context: handles type_declaration specially via type_spec."""
    curr = node.parent
    while curr:
        if curr.type in ("function_declaration", "method_declaration", "type_declaration"):
            if curr.type == "type_declaration":
                type_spec = curr.child_by_field_name("type_spec")
                if type_spec:
                    name_node = type_spec.child_by_field_name("name")
                    return (
                        ext.get_node_text(name_node) if name_node else None,
                        curr.type,
                        curr.start_point[0] + 1,
                    )
            else:
                name_node = curr.child_by_field_name("name")
                return (
                    ext.get_node_text(name_node) if name_node else None,
                    curr.type,
                    curr.start_point[0] + 1,
                )
        curr = curr.parent
    return None, None, None


# -- finder hooks ----------------------------------------------------


def find_go_functions(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find all Go function and method declarations, including receiver types."""
    functions: list[dict[str, Any]] = []
    query_str = ext.config.queries["functions"]

    captures_by_function: dict[int, dict[str, Any]] = {}

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "function_node":
            func_id = node.id
            if func_id not in captures_by_function:
                captures_by_function[func_id] = {
                    "node": node,
                    "name": None,
                    "params": None,
                    "receiver": None,
                }
        elif capture_name == "name":
            func_node = _go_find_function_node_for_name(node)
            if func_node:
                func_id = func_node.id
                if func_id not in captures_by_function:
                    captures_by_function[func_id] = {
                        "node": func_node,
                        "name": None,
                        "params": None,
                        "receiver": None,
                    }
                captures_by_function[func_id]["name"] = ext.get_node_text(node)
        elif capture_name == "params":
            func_node = _go_find_function_node_for_params(node)
            if func_node:
                func_id = func_node.id
                if func_id not in captures_by_function:
                    captures_by_function[func_id] = {
                        "node": func_node,
                        "name": None,
                        "params": None,
                        "receiver": None,
                    }
                captures_by_function[func_id]["params"] = node
        elif capture_name == "receiver":
            func_node = node.parent
            if func_node and func_node.type == "method_declaration":
                func_id = func_node.id
                if func_id not in captures_by_function:
                    captures_by_function[func_id] = {
                        "node": func_node,
                        "name": None,
                        "params": None,
                        "receiver": None,
                    }
                captures_by_function[func_id]["receiver"] = node

    for func_id, data in captures_by_function.items():
        if data["name"]:
            func_node = data["node"]
            name = data["name"]

            args: list[str] = []
            if data["params"]:
                args = _go_extract_parameters(ext, data["params"])

            receiver_type: str | None = None
            if data["receiver"]:
                receiver_type = _go_extract_receiver(ext, data["receiver"])

            context, context_type, context_line = _go_get_parent_context(ext, func_node)
            class_context = receiver_type or (
                context if context_type == "type_declaration" else None
            )

            docstring = ext.get_docstring(func_node)

            func_data: dict[str, Any] = {
                "name": name,
                "line_number": func_node.start_point[0] + 1,
                "end_line": func_node.end_point[0] + 1,
                "args": args,
                "class_context": class_context,
                "decorators": [],
                "lang": ext.language_name,
                "is_dependency": False,
            }

            if ext.index_source:
                func_data["source"] = ext.get_node_text(func_node)
                func_data["docstring"] = docstring

            functions.append(func_data)

    return functions


def find_go_classes(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find all Go struct definitions (mapped to 'classes')."""
    structs: list[dict[str, Any]] = []
    struct_query_str = ext.config.queries["structs"]
    for node, capture_name in ext.run_query(struct_query_str, root_node):
        if capture_name == "name":
            struct_node = _go_find_type_declaration_for_name(node)
            if struct_node:
                name = ext.get_node_text(node)
                class_data: dict[str, Any] = {
                    "name": name,
                    "line_number": struct_node.start_point[0] + 1,
                    "end_line": struct_node.end_point[0] + 1,
                    "bases": [],
                    "decorators": [],
                    "lang": ext.language_name,
                    "is_dependency": False,
                }
                if ext.index_source:
                    class_data["source"] = ext.get_node_text(struct_node)
                    class_data["docstring"] = ext.get_docstring(struct_node)

                structs.append(class_data)
    return structs


def find_go_interfaces(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find all Go interface definitions (extra entity type)."""
    interfaces: list[dict[str, Any]] = []
    interface_query_str = ext.config.queries["interfaces"]
    for node, capture_name in ext.run_query(interface_query_str, root_node):
        if capture_name == "name":
            interface_node = _go_find_type_declaration_for_name(node)
            if interface_node:
                name = ext.get_node_text(node)
                class_data: dict[str, Any] = {
                    "name": name,
                    "line_number": interface_node.start_point[0] + 1,
                    "end_line": interface_node.end_point[0] + 1,
                    "bases": [],
                    "decorators": [],
                    "lang": ext.language_name,
                    "is_dependency": False,
                }
                if ext.index_source:
                    class_data["source"] = ext.get_node_text(interface_node)
                    class_data["docstring"] = ext.get_docstring(interface_node)

                interfaces.append(class_data)
    return interfaces


def find_go_imports(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find all Go import declarations."""
    imports: list[dict[str, Any]] = []
    query_str = ext.config.queries["imports"]

    for node, capture_name in ext.run_query(query_str, root_node):
        line_number = node.start_point[0] + 1

        if capture_name == "path":
            path_text = ext.get_node_text(node).strip('"')
            package_name = path_text.split("/")[-1]

            alias: str | None = None
            import_spec = node.parent
            if import_spec and import_spec.type == "import_spec":
                alias_node = import_spec.child_by_field_name("name")
                if alias_node:
                    alias = ext.get_node_text(alias_node)

            imports.append(
                {
                    "name": package_name,
                    "source": path_text,
                    "alias": alias,
                    "line_number": line_number,
                    "lang": ext.language_name,
                }
            )

    return imports


def find_go_calls(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find all Go function/method calls."""
    calls: list[dict[str, Any]] = []
    query_str = ext.config.queries["calls"]

    seen_calls: set[str] = set()

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "name":
            call_node = node.parent
            while call_node and call_node.type != "call_expression":
                call_node = call_node.parent

            if call_node:
                name = ext.get_node_text(node)
                line_number = node.start_point[0] + 1

                call_key = f"{name}_{line_number}"
                if call_key in seen_calls:
                    continue
                seen_calls.add(call_key)

                full_name = (
                    ext.get_node_text(call_node.child_by_field_name("function"))
                    if call_node.child_by_field_name("function")
                    else name
                )

                # Resolve context
                context_name, context_type, context_line = _go_get_parent_context(
                    ext, node
                )

                # In Go, methods are defined on types (structs/interfaces). If we are in a method, the context is the method name.
                # Ideally we might want the receiver type as "class_context", but this requires more complex AST traversal up to the method declaration's receiver.
                # For now, we reuse the context resolution logic.
                class_context = None

                call_data: dict[str, Any] = {
                    "name": name,
                    "full_name": full_name,
                    "line_number": line_number,
                    "args": [],
                    "inferred_obj_type": None,
                    "context": (context_name, context_type, context_line),
                    "class_context": class_context,
                    "lang": ext.language_name,
                    "is_dependency": False,
                }
                calls.append(call_data)

    return calls


def find_go_variables(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find all Go variable declarations (var and short :=)."""
    variables: list[dict[str, Any]] = []
    query_str = ext.config.queries["variables"]

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "name":
            name = ext.get_node_text(node)

            variable_data: dict[str, Any] = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "value": None,
                "type": None,
                "context": None,
                "class_context": None,
                "lang": ext.language_name,
                "is_dependency": False,
            }
            variables.append(variable_data)

    return variables


# -- pre-scan --------------------------------------------------------


def pre_scan_go(files: list[Path], parser_wrapper: Any) -> dict[str, list[str]]:
    """Scans Go files to create a map of function/struct names to their file paths."""
    imports_map: dict[str, list[str]] = {}
    query_str = """
        (function_declaration name: (identifier) @name)
        (method_declaration name: (field_identifier) @name)
        (type_declaration (type_spec name: (type_identifier) @name))
    """

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
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


# ===================================================================
#  RUST HOOKS
# ===================================================================


# -- internal helpers ------------------------------------------------


def _rust_get_parent_context(
    ext: GenericExtractor, node: Any
) -> tuple[str | None, str | None, int | None]:
    """Rust-specific parent context: handles impl_item by finding the type_identifier."""
    curr = node.parent
    while curr:
        if curr.type in (
            "function_item",
            "struct_item",
            "enum_item",
            "trait_item",
            "impl_item",
        ):
            name_node = curr.child_by_field_name("name")
            if not name_node and curr.type == "impl_item":
                # For impl blocks, use the type name
                # impl Trait for Type { ... } -> we want Type
                # impl Type { ... } -> we want Type
                # Let's find the last type_identifier before the block
                for child in reversed(curr.children):
                    if child.type == "type_identifier":
                        name_node = child
                        break

            return (
                ext.get_node_text(name_node) if name_node else None,
                curr.type,
                curr.start_point[0] + 1,
            )
        curr = curr.parent
    return None, None, None


def _rust_parse_function_args(
    ext: GenericExtractor, params_node: Any
) -> list[dict[str, Any]]:
    """Helper to parse function arguments from a Rust (parameters) node."""
    args: list[dict[str, Any]] = []
    for param in params_node.named_children:
        arg_info: dict[str, Any] = {"name": "", "type": None}
        if param.type == "parameter":
            pattern_node = param.child_by_field_name("pattern")
            type_node = param.child_by_field_name("type")
            if pattern_node:
                arg_info["name"] = ext.get_node_text(pattern_node)
            if type_node:
                arg_info["type"] = ext.get_node_text(type_node)
            args.append(arg_info)
        elif param.type == "self_parameter":
            arg_info["name"] = ext.get_node_text(param)
            arg_info["type"] = "self"
            args.append(arg_info)
    return args


# -- finder hooks ----------------------------------------------------


def find_rust_functions(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find all Rust function definitions."""
    functions: list[dict[str, Any]] = []
    # Query that just finds the function items
    query_str = "(function_item) @f"

    for func_node, _ in ext.run_query(query_str, root_node):
        # Use child_by_field_name for reliable identification
        name_node = func_node.child_by_field_name("name")
        params_node = func_node.child_by_field_name("parameters")

        if name_node:
            name = ext.get_node_text(name_node)
            # Convert args to a list of strings for Neo4j property compatibility
            raw_args = (
                _rust_parse_function_args(ext, params_node) if params_node else []
            )
            params: list[str] = []
            for arg in raw_args:
                arg_str = arg["name"]
                if arg["type"]:
                    arg_str += f": {arg['type']}"
                params.append(arg_str)

            func_data: dict[str, Any] = {
                "name": name,
                "line_number": name_node.start_point[0] + 1,
                "end_line": func_node.end_point[0] + 1,
                "params": params,  # Renamed to params to match other languages
                "args": params,  # Keep args for compatibility
            }

            if ext.index_source:
                func_data["source"] = ext.get_node_text(func_node)

            functions.append(func_data)
    return functions


def find_rust_classes(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find all Rust struct, enum, and trait definitions (mapped to 'classes')."""
    structs: list[dict[str, Any]] = []
    query_str = """
    [
        (struct_item) @s
        (enum_item) @e
        (trait_item) @t
    ]
    """
    for item_node, _ in ext.run_query(query_str, root_node):
        # Find name using field name or fallback
        name_node = item_node.child_by_field_name("name")
        if not name_node:
            # Fallback: find first type_identifier
            for child in item_node.children:
                if child.type == "type_identifier":
                    name_node = child
                    break

        if name_node:
            name = ext.get_node_text(name_node)
            struct_data: dict[str, Any] = {
                "name": name,
                "line_number": name_node.start_point[0] + 1,
                "end_line": item_node.end_point[0] + 1,
                "bases": [],
            }

            if ext.index_source:
                struct_data["source"] = ext.get_node_text(item_node)

            structs.append(struct_data)
    return structs


def find_rust_traits(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find all Rust trait definitions (extra entity type)."""
    traits: list[dict[str, Any]] = []
    query_str = ext.config.queries["traits"]
    for match in ext.run_query(query_str, root_node):
        node, capture_name = match
        if capture_name == "trait_node":
            trait_node = node
            name_node = next(
                (
                    n
                    for n, c in ext.run_query(
                        "(trait_item name: (type_identifier) @name)", trait_node
                    )
                    if c == "name"
                ),
                None,
            )
            if name_node:
                name = ext.get_node_text(name_node)
                trait_data: dict[str, Any] = {
                    "name": name,
                    "line_number": name_node.start_point[0] + 1,
                    "end_line": trait_node.end_point[0] + 1,
                }

                if ext.index_source:
                    trait_data["source"] = ext.get_node_text(trait_node)

                traits.append(trait_data)
    return traits


def find_rust_imports(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find all Rust use declarations."""
    imports: list[dict[str, Any]] = []
    query_str = ext.config.queries["imports"]
    for node, _ in ext.run_query(query_str, root_node):
        full_import_name = ext.get_node_text(node)
        alias: str | None = None

        alias_match = re.search(r"as\s+(\w+)\s*;?$", full_import_name)
        if alias_match:
            alias = alias_match.group(1)
            name = alias
        else:
            cleaned_path = re.sub(r";$", "", full_import_name).strip()
            last_part = cleaned_path.split("::")[-1]
            if last_part.strip() == "*":
                name = "*"
            else:
                name_match = re.findall(r"(\w+)", last_part)
                name = name_match[-1] if name_match else last_part

        imports.append(
            {
                "name": name,
                "full_import_name": full_import_name,
                "line_number": node.start_point[0] + 1,
                "alias": alias,
            }
        )
    return imports


def find_rust_calls(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find all Rust function and method calls."""
    calls: list[dict[str, Any]] = []
    query_str = ext.config.queries["calls"]
    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "name":
            # Find the call_expression
            call_node = node.parent
            while (
                call_node
                and call_node.type != "call_expression"
                and call_node.type != "source_file"
            ):
                call_node = call_node.parent

            call_name = ext.get_node_text(node)

            # Extract arguments
            args: list[str] = []
            if call_node and call_node.type == "call_expression":
                args_node = call_node.child_by_field_name("arguments")
                if args_node:
                    for child in args_node.children:
                        if child.type not in ("(", ")", ","):
                            args.append(ext.get_node_text(child))

            calls.append(
                {
                    "name": call_name,
                    "full_name": (
                        ext.get_node_text(call_node) if call_node else call_name
                    ),
                    "line_number": node.start_point[0] + 1,
                    "args": args,
                    "context": _rust_get_parent_context(ext, node),
                    "lang": ext.language_name,
                    "is_dependency": False,
                }
            )
    return calls


# -- pre-scan --------------------------------------------------------


def pre_scan_rust(files: list[Path], parser_wrapper: Any) -> dict[str, list[str]]:
    """Scans Rust files to create a map of function/struct/enum/trait names to their file paths."""
    imports_map: dict[str, list[str]] = {}
    query_str = """
        (function_item name: (identifier) @name)
        (struct_item name: (type_identifier) @name)
        (enum_item name: (type_identifier) @name)
        (trait_item name: (type_identifier) @name)
    """

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


# ===================================================================
#  RUBY HOOKS
# ===================================================================


# -- internal helpers ------------------------------------------------


def _ruby_get_parent_context(
    ext: GenericExtractor,
    node: Any,
    types: tuple[str, ...] = ("class", "module", "method"),
) -> tuple[str | None, str | None, int | None]:
    """Find parent context for Ruby constructs."""
    curr = node.parent
    while curr:
        if curr.type in types:
            name_node = curr.child_by_field_name("name")
            if name_node:
                return (
                    ext.get_node_text(name_node),
                    curr.type,
                    curr.start_point[0] + 1,
                )
        curr = curr.parent
    return None, None, None


def _ruby_enclosing_class_name(ext: GenericExtractor, node: Any) -> str | None:
    """Get the name of the enclosing class for a Ruby node."""
    name, typ, _ = _ruby_get_parent_context(ext, node, ("class",))
    return name


def _ruby_parse_method_parameters(ext: GenericExtractor, method_node: Any) -> list[str]:
    """Parse method parameters from a Ruby method node."""
    params: list[str] = []
    # Look for parameters in the method node
    for child in method_node.children:
        if child.type == "identifier" and child != method_node.child_by_field_name(
            "name"
        ):
            # This is likely a parameter
            params.append(ext.get_node_text(child))
    return params


# -- finder hooks ----------------------------------------------------


def find_ruby_functions(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find all Ruby function/method definitions."""
    functions: list[dict[str, Any]] = []
    query_str = ext.config.queries["functions"]

    # Collect all captures first
    all_captures = list(ext.run_query(query_str, root_node))

    # Group captures by function node using a different approach
    captures_by_function: dict[int, dict[str, Any]] = {}
    for node, capture_name in all_captures:
        if capture_name == "function_node":
            captures_by_function[id(node)] = {"node": node, "name": None}

    # Now find names for each function
    for node, capture_name in all_captures:
        if capture_name == "name":
            # Find which function this name belongs to
            for func_id, func_data in captures_by_function.items():
                func_node = func_data["node"]
                # Check if this name node is within the function node
                if (
                    node.start_byte >= func_node.start_byte
                    and node.end_byte <= func_node.end_byte
                ):
                    captures_by_function[func_id]["name"] = ext.get_node_text(node)
                    break

    # Build function entries
    for func_data in captures_by_function.values():
        func_node = func_data["node"]
        name = func_data["name"]

        if name:
            args = _ruby_parse_method_parameters(ext, func_node)

            # Get context and docstring
            context, context_type, _ = _ruby_get_parent_context(ext, func_node)
            class_context = context if context_type in ("class", "module") else None
            docstring = ext.get_docstring(func_node)

            func_entry: dict[str, Any] = {
                "name": name,
                "line_number": func_node.start_point[0] + 1,
                "end_line": func_node.end_point[0] + 1,
                "args": args,
                "lang": ext.language_name,
                "is_dependency": False,
            }
            if ext.index_source:
                func_entry["source"] = ext.get_node_text(func_node)
                func_entry["docstring"] = docstring

            functions.append(func_entry)

    return functions


def find_ruby_classes(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find all Ruby class definitions."""
    classes: list[dict[str, Any]] = []
    query_str = ext.config.queries["classes"]

    # Collect all captures first
    all_captures = list(ext.run_query(query_str, root_node))

    # Group captures by class node using a different approach
    captures_by_class: dict[int, dict[str, Any]] = {}
    for node, capture_name in all_captures:
        if capture_name == "class":
            captures_by_class[id(node)] = {"node": node, "name": None}

    # Now find names for each class
    for node, capture_name in all_captures:
        if capture_name == "name":
            # Find which class this name belongs to
            for class_id, class_data in captures_by_class.items():
                class_node = class_data["node"]
                # Check if this name node is within the class node
                if (
                    node.start_byte >= class_node.start_byte
                    and node.end_byte <= class_node.end_byte
                ):
                    captures_by_class[class_id]["name"] = ext.get_node_text(node)
                    break

    # Build class entries
    for class_data in captures_by_class.values():
        class_node = class_data["node"]
        name = class_data["name"]

        if name:
            # Get superclass for inheritance (simplified)
            bases: list[str] = []

            # Get context and docstring
            context, context_type, _ = _ruby_get_parent_context(ext, class_node)
            class_context = context if context_type in ("class", "module") else None
            docstring = ext.get_docstring(class_node)

            class_entry: dict[str, Any] = {
                "name": name,
                "line_number": class_node.start_point[0] + 1,
                "end_line": class_node.end_point[0] + 1,
                "bases": bases,
                "context": context,
                "context_type": context_type,
                "class_context": class_context,
                "decorators": [],
                "lang": ext.language_name,
                "is_dependency": False,
            }
            if ext.index_source:
                class_entry["source"] = ext.get_node_text(class_node)
                class_entry["docstring"] = docstring

            classes.append(class_entry)

    return classes


def find_ruby_modules(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find all Ruby module definitions (extra entity type)."""
    modules: list[dict[str, Any]] = []
    query_str = ext.config.queries["modules"]
    # name via captures
    captures = list(ext.run_query(query_str, root_node))
    for node, cap in captures:
        if cap == "module_node":
            name: str | None = None
            for n, c in captures:
                if c == "name":
                    if (
                        n.start_byte >= node.start_byte
                        and n.end_byte <= node.end_byte
                    ):
                        name = ext.get_node_text(n)
                        break
            if name:
                module_data: dict[str, Any] = {
                    "name": name,
                    "line_number": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "lang": ext.language_name,
                    "is_dependency": False,
                }
                if ext.index_source:
                    module_data["source"] = ext.get_node_text(node)

                modules.append(module_data)
    return modules


def find_ruby_module_inclusions(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find all Ruby module inclusion statements (extra entity type)."""
    includes: list[dict[str, Any]] = []
    query_str = ext.config.queries["module_includes"]
    for node, cap in ext.run_query(query_str, root_node):
        if cap == "method":
            method_name = ext.get_node_text(node)
            if method_name != "include":
                continue
        if cap == "include_call":
            method: str | None = None
            module: str | None = None
            for n, c in ext.run_query(query_str, node):
                if c == "method":
                    method = ext.get_node_text(n)
                elif c == "module":
                    module = ext.get_node_text(n)
            if method == "include" and module:
                cls = _ruby_enclosing_class_name(ext, node)
                if cls:
                    includes.append(
                        {
                            "class": cls,
                            "module": module,
                            "line_number": node.start_point[0] + 1,
                            "lang": ext.language_name,
                            "is_dependency": False,
                        }
                    )
    return includes


def find_ruby_imports(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find all Ruby require/load statements."""
    imports: list[dict[str, Any]] = []
    query_str = ext.config.queries["imports"]

    # Collect all captures first
    all_captures = list(ext.run_query(query_str, root_node))

    # Group captures by import node using a different approach
    captures_by_import: dict[int, dict[str, Any]] = {}
    for node, capture_name in all_captures:
        if capture_name == "import":
            captures_by_import[id(node)] = {
                "node": node,
                "method_name": None,
                "path": None,
            }

    # Now find method names and paths for each import
    for node, capture_name in all_captures:
        if capture_name == "method_name":
            # Find which import this method name belongs to
            for import_id, import_data in captures_by_import.items():
                import_node = import_data["node"]
                # Check if this method name node is within the import node
                if (
                    node.start_byte >= import_node.start_byte
                    and node.end_byte <= import_node.end_byte
                ):
                    captures_by_import[import_id]["method_name"] = ext.get_node_text(
                        node
                    )
                    break
        elif capture_name == "path":
            # Find which import this path belongs to
            for import_id, import_data in captures_by_import.items():
                import_node = import_data["node"]
                # Check if this path node is within the import node
                if (
                    node.start_byte >= import_node.start_byte
                    and node.end_byte <= import_node.end_byte
                ):
                    captures_by_import[import_id]["path"] = ext.get_node_text(node)
                    break

    # Build import entries
    for import_data in captures_by_import.values():
        import_node = import_data["node"]
        method_name = import_data["method_name"]
        path = import_data["path"]

        if method_name and path:
            path = path.strip("'\"")

            # Only process require/load statements
            if method_name in ("require", "require_relative", "load"):
                imports.append(
                    {
                        "name": path,
                        "full_import_name": f"{method_name} '{path}'",
                        "line_number": import_node.start_point[0] + 1,
                        "alias": None,
                        "lang": ext.language_name,
                        "is_dependency": False,
                    }
                )

    return imports


def find_ruby_calls(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find all Ruby function and method calls."""
    calls: list[dict[str, Any]] = []
    query_str = ext.config.queries["calls"]

    # Collect all captures
    all_captures = list(ext.run_query(query_str, root_node))

    # Group by call node
    captures_by_call: dict[int, dict[str, Any]] = {}
    for node, capture_name in all_captures:
        if capture_name == "call_node":
            captures_by_call[id(node)] = {
                "node": node,
                "name": None,
                "receiver": None,
                "args": [],
            }

    for node, capture_name in all_captures:
        for call_id, call_data in captures_by_call.items():
            call_node = call_data["node"]
            if not (
                node.start_byte >= call_node.start_byte
                and node.end_byte <= call_node.end_byte
            ):
                continue

            if capture_name == "name":
                # The identifier could be part of receiver or arguments too, be careful
                # But tree-sitter structure ensures method name is distinct
                # Check if node is child 'method' of call_node
                if node == call_node.child_by_field_name("method"):
                    captures_by_call[call_id]["name"] = ext.get_node_text(node)

            elif capture_name == "receiver":
                captures_by_call[call_id]["receiver"] = ext.get_node_text(node)

            elif capture_name == "args":
                # Capture arguments
                args_text = ext.get_node_text(node)
                # Simple heuristic: split by comma
                captures_by_call[call_id]["args"] = [
                    a.strip()
                    for a in args_text.strip("()").split(",")
                    if a.strip()
                ]

    for call_data in captures_by_call.values():
        call_node = call_data["node"]
        name = call_data["name"]

        if name:
            receiver = call_data["receiver"]
            full_name = f"{receiver}.{name}" if receiver else name

            context_name, context_type, context_line = _ruby_get_parent_context(
                ext, call_node
            )
            class_context = (
                context_name if context_type in ("class", "module") else None
            )
            if context_type == "method":
                # If inside a method, try to find enclosing class too
                enclosing_class, _, _ = _ruby_get_parent_context(
                    ext, call_node.parent, ("class", "module")
                )
                class_context = enclosing_class

            calls.append(
                {
                    "name": name,
                    "full_name": full_name,
                    "line_number": call_node.start_point[0] + 1,
                    "args": call_data["args"],
                    "inferred_obj_type": None,
                    "context": (context_name, context_type, context_line),
                    "class_context": class_context,
                    "lang": ext.language_name,
                    "is_dependency": False,
                }
            )

    return calls


def find_ruby_variables(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find all Ruby variable assignments."""
    variables: list[dict[str, Any]] = []
    query_str = ext.config.queries["variables"]

    # Group captures by assignment node
    captures_by_assignment: dict[int, dict[str, Any]] = {}
    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "name":
            # Find the parent assignment node
            current = node.parent
            while current and current.type != "assignment":
                current = current.parent
            if current:
                assignment_id = id(current)
                if assignment_id not in captures_by_assignment:
                    captures_by_assignment[assignment_id] = {
                        "node": current,
                        "name": None,
                        "value": None,
                    }
                captures_by_assignment[assignment_id]["name"] = ext.get_node_text(node)
        elif capture_name == "value":
            # Find the parent assignment node
            current = node.parent
            while current and current.type != "assignment":
                current = current.parent
            if current:
                assignment_id = id(current)
                if assignment_id not in captures_by_assignment:
                    captures_by_assignment[assignment_id] = {
                        "node": current,
                        "name": None,
                        "value": None,
                    }
                captures_by_assignment[assignment_id]["value"] = ext.get_node_text(node)

    # Build variable entries
    for var_data in captures_by_assignment.values():
        name = var_data["name"]
        value = var_data["value"]

        if name:
            # Determine variable type based on name prefix
            var_type = "local"
            if name.startswith("@"):
                var_type = "instance"
            elif name.startswith("@@"):
                var_type = "class"
            elif name.startswith("$"):
                var_type = "global"

            variables.append(
                {
                    "name": name,
                    "line_number": var_data["node"].start_point[0] + 1,
                    "value": value,
                    "type": var_type,
                    "context": None,  # Placeholder
                    "class_context": None,  # Placeholder
                    "lang": ext.language_name,
                    "is_dependency": False,
                }
            )

    return variables


# -- pre-scan --------------------------------------------------------


def pre_scan_ruby(files: list[Path], parser_wrapper: Any) -> dict[str, list[str]]:
    """Scans Ruby files to create a map of class/method names to their file paths."""
    imports_map: dict[str, list[str]] = {}
    query_str = """
        (class
            name: (constant) @name
        )
        (module
            name: (constant) @name
        )
        (method
            name: (identifier) @name
        )
    """

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
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


# ===================================================================
#  JAVA HOOKS
# ===================================================================


# -- internal helpers ------------------------------------------------


def _java_get_parent_context(
    ext: GenericExtractor, node: Any
) -> tuple[str | None, str | None, int | None]:
    """Java-specific parent context resolution."""
    curr = node.parent
    while curr:
        if curr.type in ("method_declaration", "constructor_declaration"):
            name_node = curr.child_by_field_name("name")
            return (
                ext.get_node_text(name_node) if name_node else None,
                curr.type,
                curr.start_point[0] + 1,
            )
        if curr.type in (
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "annotation_type_declaration",
        ):
            name_node = curr.child_by_field_name("name")
            return (
                ext.get_node_text(name_node) if name_node else None,
                curr.type,
                curr.start_point[0] + 1,
            )
        curr = curr.parent
    return None, None, None


def _java_extract_parameter_names(params_text: str) -> list[str]:
    """Extract parameter names from a Java formal_parameters text representation."""
    params: list[str] = []
    if not params_text or params_text.strip() == "()":
        return params

    params_content = params_text.strip("()")
    if not params_content:
        return params

    for param in params_content.split(","):
        param = param.strip()
        if param:
            parts = param.split()
            if len(parts) >= 2:
                param_name = parts[-1]
                params.append(param_name)

    return params


# -- finder hooks ----------------------------------------------------


def find_java_functions(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find all Java method and constructor declarations."""
    functions: list[dict[str, Any]] = []
    query_str = ext.config.queries["functions"]
    captures = list(ext.run_query(query_str, root_node))

    # Group by node identity or stable key to avoid duplicates
    seen_nodes: set[tuple[int, int, str]] = set()

    for node, capture_name in captures:
        if capture_name == "function_node":
            node_id = (node.start_byte, node.end_byte, node.type)
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)

            try:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                name_node = node.child_by_field_name("name")
                if name_node:
                    func_name = ext.get_node_text(name_node)

                    params_node = node.child_by_field_name("parameters")
                    parameters: list[str] = []
                    if params_node:
                        params_text = ext.get_node_text(params_node)
                        parameters = _java_extract_parameter_names(params_text)

                    source_text = ext.get_node_text(node)

                    # Get class context
                    context_name, context_type, context_line = (
                        _java_get_parent_context(ext, node)
                    )

                    func_data: dict[str, Any] = {
                        "name": func_name,
                        "parameters": parameters,
                        "line_number": start_line,
                        "end_line": end_line,
                        "path": ext.current_path,
                        "lang": ext.language_name,
                        "context": context_name,
                        "class_context": (
                            context_name
                            if context_type and "class" in context_type
                            else None
                        ),
                    }

                    if ext.index_source:
                        func_data["source"] = source_text

                    functions.append(func_data)

            except Exception as e:
                error_logger(
                    f"Error parsing function in {ext.current_path}: {e}"
                )
                continue

    return functions


def find_java_classes(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find all Java class, interface, enum, and annotation type declarations."""
    classes: list[dict[str, Any]] = []
    query_str = ext.config.queries["classes"]
    captures = list(ext.run_query(query_str, root_node))

    seen_nodes: set[tuple[int, int, str]] = set()

    for node, capture_name in captures:
        if capture_name == "class":
            node_id = (node.start_byte, node.end_byte, node.type)
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)

            try:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                name_node = node.child_by_field_name("name")
                if name_node:
                    class_name = ext.get_node_text(name_node)
                    source_text = ext.get_node_text(node)

                    bases: list[str] = []
                    # Look for superclass (extends)
                    superclass_node = node.child_by_field_name("superclass")
                    if superclass_node:
                        # In Java, superclass field usually points to a type node
                        bases.append(ext.get_node_text(superclass_node))

                    # Look for super_interfaces (implements)
                    interfaces_node = node.child_by_field_name("interfaces")
                    if not interfaces_node:
                        interfaces_node = next(
                            (
                                c
                                for c in node.children
                                if c.type == "super_interfaces"
                            ),
                            None,
                        )

                    if interfaces_node:
                        type_list = interfaces_node.child_by_field_name("list")
                        if not type_list:
                            type_list = next(
                                (
                                    c
                                    for c in interfaces_node.children
                                    if c.type == "type_list"
                                ),
                                None,
                            )

                        if type_list:
                            for child in type_list.children:
                                if child.type in (
                                    "type_identifier",
                                    "generic_type",
                                    "scoped_type_identifier",
                                ):
                                    bases.append(ext.get_node_text(child))
                        else:
                            for child in interfaces_node.children:
                                if child.type in (
                                    "type_identifier",
                                    "generic_type",
                                    "scoped_type_identifier",
                                ):
                                    bases.append(ext.get_node_text(child))

                    class_data: dict[str, Any] = {
                        "name": class_name,
                        "line_number": start_line,
                        "end_line": end_line,
                        "bases": bases,
                        "path": ext.current_path,
                        "lang": ext.language_name,
                    }

                    if ext.index_source:
                        class_data["source"] = source_text

                    classes.append(class_data)

            except Exception as e:
                error_logger(
                    f"Error parsing class in {ext.current_path}: {e}"
                )
                continue

    return classes


def find_java_imports(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find all Java import declarations."""
    imports: list[dict[str, Any]] = []
    query_str = ext.config.queries["imports"]
    captures = list(ext.run_query(query_str, root_node))

    for node, capture_name in captures:
        if capture_name == "import":
            try:
                import_text = ext.get_node_text(node)
                import_match = re.search(
                    r"import\s+(?:static\s+)?([^;]+)", import_text
                )
                if import_match:
                    import_path = import_match.group(1).strip()

                    import_data: dict[str, Any] = {
                        "name": import_path,
                        "full_import_name": import_path,
                        "line_number": node.start_point[0] + 1,
                        "alias": None,
                        "context": (None, None),
                        "lang": ext.language_name,
                        "is_dependency": False,
                    }
                    imports.append(import_data)
            except Exception as e:
                error_logger(f"Error parsing import: {e}")
                continue

    return imports


def find_java_calls(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Find all Java method invocations and object creation expressions."""
    calls: list[dict[str, Any]] = []
    seen_calls: set[str] = set()
    query_str = ext.config.queries["calls"]
    captures = list(ext.run_query(query_str, root_node))

    debug_log(f"Processing {len(captures)} captures for calls")

    for node, capture_name in captures:
        if capture_name == "name":
            try:
                call_name = ext.get_node_text(node)
                line_number = node.start_point[0] + 1

                # Ensure we identify the full call node
                call_node = node.parent
                while call_node and call_node.type not in (
                    "method_invocation",
                    "object_creation_expression",
                ):
                    call_node = call_node.parent

                if not call_node:
                    # fallback if we matched a loose identifier
                    call_node = node

                # Avoid duplicates
                call_key = f"{call_name}_{line_number}"
                if call_key in seen_calls:
                    continue
                seen_calls.add(call_key)

                # Extract arguments
                args: list[str] = []
                if call_node:
                    args_node = next(
                        (
                            c
                            for c in call_node.children
                            if c.type == "argument_list"
                        ),
                        None,
                    )
                    if args_node:
                        for arg in args_node.children:
                            if arg.type not in ("(", ")", ","):
                                args.append(ext.get_node_text(arg))

                # Extract meaningful full_name
                full_name = call_name
                if call_node.type == "method_invocation":
                    obj_node = call_node.child_by_field_name("object")
                    if obj_node:
                        full_name = f"{ext.get_node_text(obj_node)}.{call_name}"
                elif call_node.type == "object_creation_expression":
                    type_node = call_node.child_by_field_name("type")
                    if type_node:
                        full_name = ext.get_node_text(type_node)

                ctx_name, ctx_type, ctx_line = _java_get_parent_context(ext, node)

                debug_log(
                    f"Found call: {call_name} (full_name: {full_name}, args: {args}) in context {ctx_name}"
                )

                call_data: dict[str, Any] = {
                    "name": call_name,
                    "full_name": full_name,
                    "line_number": line_number,
                    "args": args,
                    "inferred_obj_type": None,
                    "context": (ctx_name, ctx_type, ctx_line),
                    "class_context": (
                        (ctx_name, ctx_line)
                        if ctx_type and "class" in ctx_type
                        else (None, None)
                    ),
                    "lang": ext.language_name,
                    "is_dependency": False,
                }
                calls.append(call_data)
            except Exception as e:
                error_logger(f"Error parsing call: {e}")
                continue

    return calls


def find_java_variables(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find all Java variable declarations (local and field)."""
    variables: list[dict[str, Any]] = []
    query_str = ext.config.queries["variables"]
    captures = list(ext.run_query(query_str, root_node))

    seen_vars: set[int] = set()

    # Re-approach: Iterate captures and collect finding.
    # Tree sitter returns a list of (node, capture_name).

    # Simpler approach: Iterate 'name' captures that are inside a variable declaration context

    for node, capture_name in captures:
        if capture_name == "name":
            # Check parent to confirm it's a variable declarator
            if node.parent.type == "variable_declarator":
                var_name = ext.get_node_text(node)
                start_line = node.start_point[0] + 1

                # Get type? Type is sibling of declarator usually, or child of declaration
                # local_variable_declaration -> type, variable_declarator
                declaration = node.parent.parent
                type_node = declaration.child_by_field_name("type")
                var_type = ext.get_node_text(type_node) if type_node else "Unknown"

                start_byte = node.start_byte
                if start_byte in seen_vars:
                    continue
                seen_vars.add(start_byte)

                ctx_name, ctx_type, ctx_line = _java_get_parent_context(ext, node)

                variables.append(
                    {
                        "name": var_name,
                        "type": var_type,
                        "line_number": start_line,
                        "path": ext.current_path,
                        "lang": ext.language_name,
                        "context": ctx_name,
                        "class_context": (
                            ctx_name
                            if ctx_type and "class" in ctx_type
                            else None
                        ),
                    }
                )

    return variables


# -- pre-scan --------------------------------------------------------


def pre_scan_java(files: list[Path], parser_wrapper: Any) -> dict[str, list[str]]:
    """Scans Java files to create a map of class/interface names to their file paths.

    Uses regex-based scanning (not tree-sitter) for performance.
    """
    name_to_files: dict[str, list[str]] = {}

    for path in files:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            class_matches = re.finditer(
                r"\b(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:abstract\s+)?(?:final\s+)?class\s+(\w+)",
                content,
            )
            for match in class_matches:
                class_name = match.group(1)
                if class_name not in name_to_files:
                    name_to_files[class_name] = []
                name_to_files[class_name].append(str(path))

            interface_matches = re.finditer(
                r"\b(?:public\s+|private\s+|protected\s+)?interface\s+(\w+)",
                content,
            )
            for match in interface_matches:
                interface_name = match.group(1)
                if interface_name not in name_to_files:
                    name_to_files[interface_name] = []
                name_to_files[interface_name].append(str(path))

        except Exception as e:
            error_logger(f"Error pre-scanning Java file {path}: {e}")

    return name_to_files


# ===========================================================================
# C  -- helper functions
# ===========================================================================


def _c_get_parent_context(
    ext: GenericExtractor,
    node: Any,
    types: tuple[str, ...] = (
        "function_definition",
        "struct_specifier",
        "union_specifier",
        "enum_specifier",
    ),
) -> tuple[str | None, str | None, int | None]:
    """Get parent context for nested C constructs.

    Unlike the generic ``get_parent_context``, this traverses the declarator
    chain for ``function_definition`` nodes to locate the actual identifier.
    """
    curr = node.parent
    while curr:
        if curr.type in types:
            if curr.type == "function_definition":
                # Traverse declarator to find name and use its line number
                decl = curr.child_by_field_name("declarator")
                while decl:
                    if decl.type == "identifier":
                        return (
                            ext.get_node_text(decl),
                            curr.type,
                            decl.start_point[0] + 1,
                        )

                    # Handle recursive declarators (function, pointer, array, parenthesized)
                    child = decl.child_by_field_name("declarator")
                    if child:
                        decl = child
                    else:
                        # Fallback if structure is different
                        break
            else:
                name_node = curr.child_by_field_name("name")
                if name_node:
                    return (
                        ext.get_node_text(name_node),
                        curr.type,
                        name_node.start_point[0] + 1,
                    )
        curr = curr.parent
    return None, None, None


def _c_calculate_complexity(ext: GenericExtractor, node: Any) -> int:
    """Calculate cyclomatic complexity for C functions."""
    complexity_nodes = {
        "if_statement",
        "for_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
        "case_statement",
        "conditional_expression",
        "logical_expression",
        "binary_expression",
        "goto_statement",
    }
    count = 1

    def traverse(n: Any) -> None:
        nonlocal count
        if n.type in complexity_nodes:
            count += 1
        for child in n.children:
            traverse(child)

    traverse(node)
    return count


def _c_get_docstring(ext: GenericExtractor, node: Any) -> str | None:
    """Extract comments as documentation for C nodes."""
    # Look for comments before the node
    if node.parent:
        for child in node.parent.children:
            if child.type == "comment" and child.start_point[0] < node.start_point[0]:
                return ext.get_node_text(child)
    return None


def _c_parse_function_args(
    ext: GenericExtractor, params_node: Any
) -> list[dict[str, Any]]:
    """Enhanced helper to parse function arguments from a (parameter_list) node."""
    args: list[dict[str, Any]] = []
    if not params_node:
        return args

    for param in params_node.named_children:
        if param.type == "parameter_declaration":
            arg_info: dict[str, Any] = {
                "name": "",
                "type": None,
                "is_pointer": False,
                "is_array": False,
            }

            # Find the declarator (variable name)
            declarator = param.child_by_field_name("declarator")
            if declarator:
                if declarator.type == "identifier":
                    arg_info["name"] = ext.get_node_text(declarator)
                elif declarator.type == "pointer_declarator":
                    arg_info["is_pointer"] = True
                    inner_declarator = declarator.child_by_field_name("declarator")
                    if inner_declarator and inner_declarator.type == "identifier":
                        arg_info["name"] = ext.get_node_text(inner_declarator)
                elif declarator.type == "array_declarator":
                    arg_info["is_array"] = True
                    inner_declarator = declarator.child_by_field_name("declarator")
                    if inner_declarator and inner_declarator.type == "identifier":
                        arg_info["name"] = ext.get_node_text(inner_declarator)

            # Find the type
            type_node = param.child_by_field_name("type")
            if type_node:
                arg_info["type"] = ext.get_node_text(type_node)

            # Handle variadic arguments
            if param.type == "variadic_parameter":
                arg_info["name"] = "..."
                arg_info["type"] = "variadic"

            args.append(arg_info)
    return args


# ===========================================================================
# C  -- finder hooks
# ===========================================================================


def find_c_functions(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find C function definitions."""
    functions: list[dict[str, Any]] = []
    query_str = ext.config.queries["functions"]
    for match in ext.run_query(query_str, root_node):
        capture_name = match[1]
        node = match[0]
        if capture_name == "name":
            func_node = node.parent.parent.parent
            name = ext.get_node_text(node)

            # Find parameters
            params_node = None
            body_node = None
            for child in func_node.children:
                if child.type == "function_declarator":
                    params_node = child.child_by_field_name("parameters")
                elif child.type == "compound_statement":
                    body_node = child

            args = _c_parse_function_args(ext, params_node) if params_node else []
            context, context_type, _ = _c_get_parent_context(ext, func_node)

            func_data: dict[str, Any] = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "end_line": func_node.end_point[0] + 1,
                "args": [
                    arg["name"] for arg in args if arg["name"]
                ],  # Simplified args for compatibility
                "docstring": _c_get_docstring(ext, func_node),
                "cyclomatic_complexity": _c_calculate_complexity(ext, func_node),
                "context": context,
                "context_type": context_type,
                "class_context": None,
                "decorators": [],
                "lang": ext.language_name,
                "is_dependency": False,
                "detailed_args": args,  # Keep detailed args for future use
            }

            if ext.index_source:
                func_data["source"] = ext.get_node_text(func_node)

            functions.append(func_data)
    return functions


def find_c_classes(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find structs, unions, and enums (treated as classes in C)."""
    classes: list[dict[str, Any]] = []

    # Find structs
    query_str = ext.config.queries["structs"]
    for match in ext.run_query(query_str, root_node):
        capture_name = match[1]
        node = match[0]
        if capture_name == "name":
            struct_node = node.parent
            name = ext.get_node_text(node)
            context, context_type, _ = _c_get_parent_context(ext, struct_node)

            struct_data: dict[str, Any] = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "end_line": struct_node.end_point[0] + 1,
                "bases": [],  # C doesn't have inheritance
                "docstring": _c_get_docstring(ext, struct_node),
                "context": context,
                "decorators": [],
                "lang": ext.language_name,
                "is_dependency": False,
                "type": "struct",
            }

            if ext.index_source:
                struct_data["source"] = ext.get_node_text(struct_node)

            classes.append(struct_data)

    # Find unions
    query_str = ext.config.queries["unions"]
    for match in ext.run_query(query_str, root_node):
        capture_name = match[1]
        node = match[0]
        if capture_name == "name":
            union_node = node.parent
            name = ext.get_node_text(node)
            context, context_type, _ = _c_get_parent_context(ext, union_node)

            union_data: dict[str, Any] = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "end_line": union_node.end_point[0] + 1,
                "bases": [],
                "docstring": _c_get_docstring(ext, union_node),
                "context": context,
                "decorators": [],
                "lang": ext.language_name,
                "is_dependency": False,
                "type": "union",
            }

            if ext.index_source:
                union_data["source"] = ext.get_node_text(union_node)

            classes.append(union_data)

    # Find enums
    query_str = ext.config.queries["enums"]
    for match in ext.run_query(query_str, root_node):
        capture_name = match[1]
        node = match[0]
        if capture_name == "name":
            enum_node = node.parent
            name = ext.get_node_text(node)
            context, context_type, _ = _c_get_parent_context(ext, enum_node)

            enum_data: dict[str, Any] = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "end_line": enum_node.end_point[0] + 1,
                "bases": [],
                "docstring": _c_get_docstring(ext, enum_node),
                "context": context,
                "decorators": [],
                "lang": ext.language_name,
                "is_dependency": False,
                "type": "enum",
            }

            if ext.index_source:
                enum_data["source"] = ext.get_node_text(enum_node)

            classes.append(enum_data)

    return classes


def find_c_macros(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Enhanced preprocessor macro detection for C."""
    macros: list[dict[str, Any]] = []
    query_str = ext.config.queries["macros"]
    for match in ext.run_query(query_str, root_node):
        capture_name = match[1]
        node = match[0]
        if capture_name == "name":
            macro_node = node.parent
            name = ext.get_node_text(node)

            # Extract macro value
            value = None
            if macro_node.child_by_field_name("value"):
                value = ext.get_node_text(macro_node.child_by_field_name("value"))

            # Extract parameters for function-like macros
            params: list[str] = []
            if macro_node.child_by_field_name("parameters"):
                params_node = macro_node.child_by_field_name("parameters")
                for child in params_node.children:
                    if child.type == "identifier":
                        params.append(ext.get_node_text(child))

            context, context_type, _ = _c_get_parent_context(ext, macro_node)

            macro_data: dict[str, Any] = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "end_line": macro_node.end_point[0] + 1,
                "value": value,
                "params": params,
                "context": context,
                "lang": ext.language_name,
                "is_dependency": False,
            }

            if ext.index_source:
                macro_data["source"] = ext.get_node_text(macro_node)

            macros.append(macro_data)
    return macros


def find_c_imports(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find C #include imports."""
    imports: list[dict[str, Any]] = []
    query_str = ext.config.queries["imports"]
    for match in ext.run_query(query_str, root_node):
        capture_name = match[1]
        node = match[0]
        if capture_name == "path":
            path = ext.get_node_text(node).strip('"<>')
            context, context_type, _ = _c_get_parent_context(ext, node)

            imports.append(
                {
                    "name": path,
                    "full_import_name": path,
                    "line_number": node.start_point[0] + 1,
                    "alias": None,
                    "context": context,
                    "lang": ext.language_name,
                    "is_dependency": False,
                }
            )
    return imports


def find_c_calls(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Enhanced function call detection for C."""
    calls: list[dict[str, Any]] = []
    query_str = ext.config.queries["calls"]
    for match in ext.run_query(query_str, root_node):
        capture_name = match[1]
        node = match[0]
        if capture_name == "name":
            call_node = (
                node.parent
                if node.parent.type == "call_expression"
                else node.parent.parent
            )
            call_name = ext.get_node_text(node)

            # Extract arguments
            args: list[str] = []
            args_node = call_node.child_by_field_name("arguments")
            if args_node:
                for child in args_node.children:
                    if child.type not in ["(", ")", ","]:
                        args.append(ext.get_node_text(child))

            context_name, context_type, context_line = _c_get_parent_context(
                ext, call_node
            )

            calls.append(
                {
                    "name": call_name,
                    "full_name": call_name,  # For C, function name is the same as full name
                    "line_number": node.start_point[0] + 1,
                    "args": args,
                    "inferred_obj_type": None,
                    "context": (context_name, context_type, context_line),
                    "class_context": None,
                    "lang": ext.language_name,
                    "is_dependency": False,
                }
            )
    return calls


def find_c_variables(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Enhanced variable declaration detection for C."""
    variables: list[dict[str, Any]] = []
    query_str = ext.config.queries["variables"]
    for match in ext.run_query(query_str, root_node):
        capture_name = match[1]
        node = match[0]
        if capture_name == "name":
            var_name = ext.get_node_text(node)

            # Find the declaration node
            decl_node = node.parent
            while decl_node and decl_node.type != "declaration":
                decl_node = decl_node.parent

            # Extract type information
            var_type = None
            is_pointer = False
            is_array = False
            value = None

            if decl_node:
                # Find type
                for child in decl_node.children:
                    if child.type in [
                        "primitive_type",
                        "type_identifier",
                        "sized_type_specifier",
                    ]:
                        var_type = ext.get_node_text(child)
                    elif child.type == "init_declarator":
                        # Check for pointer/array
                        if child.child_by_field_name("declarator"):
                            declarator = child.child_by_field_name("declarator")
                            if declarator.type == "pointer_declarator":
                                is_pointer = True
                            elif declarator.type == "array_declarator":
                                is_array = True

                        # Check for initial value
                        if child.child_by_field_name("value"):
                            value = ext.get_node_text(
                                child.child_by_field_name("value")
                            )

            context, context_type, _ = _c_get_parent_context(ext, node)
            class_context, _, _ = _c_get_parent_context(
                ext,
                node,
                types=("struct_specifier", "union_specifier", "enum_specifier"),
            )

            variables.append(
                {
                    "name": var_name,
                    "line_number": node.start_point[0] + 1,
                    "value": value,
                    "type": var_type,
                    "context": context,
                    "class_context": class_context,
                    "lang": ext.language_name,
                    "is_dependency": False,
                    "is_pointer": is_pointer,
                    "is_array": is_array,
                }
            )
    return variables


# ===========================================================================
# C  -- pre-scan
# ===========================================================================


def pre_scan_c(files: list[Path], parser_wrapper: Any) -> dict[str, list[str]]:
    """Scans C files to create a map of function/struct/union/enum names to their file paths."""
    imports_map: dict[str, list[str]] = {}
    query_str = """
        (function_definition
            declarator: (function_declarator
                declarator: (identifier) @name
            )
        )

        (function_definition
            declarator: (function_declarator
                declarator: (pointer_declarator
                    declarator: (identifier) @name
                )
            )
        )

        (struct_specifier
            name: (type_identifier) @name
        )

        (union_specifier
            name: (type_identifier) @name
        )

        (enum_specifier
            name: (type_identifier) @name
        )

        (type_definition
            declarator: (type_identifier) @name
        )

        (preproc_def
            name: (identifier) @name
        )
    """

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


# ===========================================================================
# C++  -- helper functions
# ===========================================================================


def _cpp_get_parent_context(
    ext: GenericExtractor,
    node: Any,
    types: tuple[str, ...] = ("function_definition", "class_definition"),
) -> tuple[str | None, str | None, int | None]:
    """Get parent context for C++ constructs.

    Traverses the declarator chain for ``function_definition`` nodes.
    """
    curr = node.parent
    while curr:
        if curr.type in types:
            if curr.type == "function_definition":
                # Traverse declarator to find name
                decl = curr.child_by_field_name("declarator")
                while decl:
                    if decl.type == "identifier":
                        return (
                            ext.get_node_text(decl),
                            curr.type,
                            decl.start_point[0] + 1,
                        )

                    child = decl.child_by_field_name("declarator")
                    if child:
                        decl = child
                    else:
                        break
                # Fallback or if not found
                return None, curr.type, curr.start_point[0] + 1
            else:
                name_node = curr.child_by_field_name("name")
                return (
                    ext.get_node_text(name_node) if name_node else None,
                    curr.type,
                    curr.start_point[0] + 1,
                )
        curr = curr.parent
    return None, None, None


def _cpp_extract_function_params(ext: GenericExtractor, func_node: Any) -> list[str]:
    """Extract parameters from a C++ function definition node.

    Unwraps pointer/reference declarators to find the underlying identifier.
    """
    params: list[str] = []
    declarator_node = func_node.child_by_field_name("declarator")
    if not declarator_node:
        return []

    parameters_node = declarator_node.child_by_field_name("parameters")
    if not parameters_node or parameters_node.type != "parameter_list":
        return []

    for param in parameters_node.children:
        if param.type == "parameter_declaration":
            # Extract name
            param_decl = param.child_by_field_name("declarator")
            # Unwrap pointers/refs to find identifier
            while param_decl and param_decl.type not in (
                "identifier",
                "field_identifier",
                "type_identifier",
            ):
                child = param_decl.child_by_field_name("declarator")
                if child:
                    param_decl = child
                else:
                    break

            name = ext.get_node_text(param_decl) if param_decl else ""

            # Extract type
            param_type_node = param.child_by_field_name("type")
            type_str = ext.get_node_text(param_type_node) if param_type_node else ""

            if name:
                # Storing "type name" string, or just name?
                # Standard in this project seems to be list of strings.
                # Given C++ complexity, providing "type name" provides more info.
                if type_str:
                    params.append(f"{type_str} {name}")
                else:
                    params.append(name)
    return params


def _cpp_get_full_name(ext: GenericExtractor, node: Any) -> str | None:
    """Builds a fully qualified name for a function or call node.

    Walks up the AST collecting namespace::class scopes.
    """
    name_parts: list[str] = []

    # Move upward and collect parent scopes
    curr = node
    while curr:
        if curr.type in ("function_definition", "function_declarator"):
            id_node = curr.child_by_field_name("declarator")
            if id_node and id_node.type == "identifier":
                name_parts.insert(0, id_node.text.decode("utf8"))
        elif curr.type == "class_specifier":
            name_node = curr.child_by_field_name("name")
            if name_node:
                name_parts.insert(0, name_node.text.decode("utf8"))
        elif curr.type == "namespace_definition":
            name_node = curr.child_by_field_name("name")
            if name_node:
                name_parts.insert(0, name_node.text.decode("utf8"))
        curr = curr.parent

    return "::".join(name_parts) if name_parts else None


# ===========================================================================
# C++  -- finder hooks
# ===========================================================================


def find_cpp_functions(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find C++ function definitions."""
    functions: list[dict[str, Any]] = []
    query_str = ext.config.queries["functions"]
    for match in ext.run_query(query_str, root_node):
        capture_name = match[1]
        node = match[0]
        if capture_name == "name":
            # node is identifier
            # node.parent is function_declarator
            # node.parent.parent is function_definition
            func_node = node.parent.parent

            # Double check to prevent crashes if AST is different (e.g. pointers)
            if func_node.type != "function_definition":
                # Fallback or try finding function_definition upwards
                curr = node
                while curr and curr.type != "function_definition":
                    curr = curr.parent
                func_node = curr

            if not func_node:
                continue

            name = ext.get_node_text(node)

            params = _cpp_extract_function_params(ext, func_node)

            func_data: dict[str, Any] = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "end_line": func_node.end_point[0] + 1,
                "args": params,
            }

            if ext.index_source:
                func_data["source"] = ext.get_node_text(func_node)

            functions.append(func_data)
    return functions


def find_cpp_classes(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find C++ class definitions."""
    classes: list[dict[str, Any]] = []
    query_str = ext.config.queries["classes"]
    for match in ext.run_query(query_str, root_node):
        capture_name = match[1]
        node = match[0]
        if capture_name == "name":
            class_node = node.parent
            name = ext.get_node_text(node)
            class_data: dict[str, Any] = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "end_line": class_node.end_point[0] + 1,
                "bases": [],  # Placeholder
            }
            if ext.index_source:
                class_data["source"] = ext.get_node_text(class_node)
            classes.append(class_data)
    return classes


def find_cpp_imports(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find C++ #include imports."""
    imports: list[dict[str, Any]] = []
    query_str = ext.config.queries["imports"]
    for match in ext.run_query(query_str, root_node):
        capture_name = match[1]
        node = match[0]
        if capture_name == "path":
            path = ext.get_node_text(node).strip("<>")
            imports.append(
                {
                    "name": path,
                    "full_import_name": path,
                    "line_number": node.start_point[0] + 1,
                    "alias": None,
                }
            )
    return imports


def find_cpp_enums(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find C++ enum definitions."""
    enums: list[dict[str, Any]] = []
    query_str = ext.config.queries["enums"]
    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "name":
            name = ext.get_node_text(node)
            enum_node = node.parent
            enum_data: dict[str, Any] = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "end_line": enum_node.end_point[0] + 1,
            }
            if ext.index_source:
                enum_data["source"] = ext.get_node_text(enum_node)
            enums.append(enum_data)
    return enums


def find_cpp_structs(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find C++ struct definitions."""
    structs: list[dict[str, Any]] = []
    query_str = ext.config.queries["structs"]
    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "name":
            name = ext.get_node_text(node)
            struct_node = node.parent
            struct_data: dict[str, Any] = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "end_line": struct_node.end_point[0] + 1,
            }
            if ext.index_source:
                struct_data["source"] = ext.get_node_text(struct_node)
            structs.append(struct_data)
    return structs


def find_cpp_unions(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find C++ union definitions."""
    unions: list[dict[str, Any]] = []
    query_str = ext.config.queries["unions"]
    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "name":
            name = ext.get_node_text(node)
            union_node = node.parent
            union_data: dict[str, Any] = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "end_line": union_node.end_point[0] + 1,
            }
            if ext.index_source:
                union_data["source"] = ext.get_node_text(union_node)
            unions.append(union_data)
    return unions


def find_cpp_macros(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find C++ preprocessor macro definitions."""
    macros: list[dict[str, Any]] = []
    query_str = ext.config.queries["macros"]
    for match in ext.run_query(query_str, root_node):
        capture_name = match[1]
        node = match[0]
        if capture_name == "name":
            macro_node = node.parent
            name = ext.get_node_text(node)
            macro_data: dict[str, Any] = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "end_line": macro_node.end_point[0] + 1,
            }
            if ext.index_source:
                macro_data["source"] = ext.get_node_text(macro_node)
            macros.append(macro_data)
    return macros


def find_cpp_lambda_assignments(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find C++ lambda expressions assigned to variables (treated as functions)."""
    functions: list[dict[str, Any]] = []
    query_str = ext.config.queries.get("lambda_assignments")
    if not query_str:
        return []

    for match in ext.run_query(query_str, root_node):
        capture_name = match[1]
        node = match[0]

        if capture_name != "name":
            continue
        assignment_node = node.parent
        lambda_node = assignment_node.child_by_field_name("value")
        if lambda_node is None or lambda_node.type != "lambda_expression":
            continue

        params_node = lambda_node.child_by_field_name("declarator")
        if params_node:
            params_node = params_node.child_by_field_name("parameters")
            name = ext.get_node_text(node)
            params_node = lambda_node.child_by_field_name("parameters")

            context, context_type, _ = _cpp_get_parent_context(ext, assignment_node)
            class_context, _, _ = _cpp_get_parent_context(
                ext, assignment_node, types=("class_definition",)
            )

            func_data: dict[str, Any] = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "end_line": assignment_node.end_point[0] + 1,
                "args": [
                    p
                    for p in [
                        ext.get_node_text(p)
                        for p in params_node.children
                        if p.type == "identifier"
                    ]
                    if p
                ]
                if params_node
                else [],
                "docstring": None,
                "cyclomatic_complexity": 1,
                "context": context,
                "context_type": context_type,
                "class_context": class_context,
                "decorators": [],
                "lang": ext.language_name,
                "is_dependency": False,
            }

            if ext.index_source:
                func_data["source"] = ext.get_node_text(assignment_node)

            functions.append(func_data)
    return functions


def find_cpp_calls(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find C++ function calls."""
    calls: list[dict[str, Any]] = []
    query_str = ext.config.queries["calls"]
    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "function_name":
            func_name = ext.get_node_text(node)
            func_node = node.parent.parent  # function_declarator -> function_definition
            full_name = _cpp_get_full_name(ext, func_node) or func_name

            # Find return type node (captured separately)
            return_type_node = None
            for n, cap in ext.run_query(query_str, func_node):
                if cap == "return_type":
                    return_type_node = n
                    break
            return_type = ext.get_node_text(return_type_node) if return_type_node else None

            # Extract parameters
            args: list[dict[str, str | None]] = []
            parameters_node = func_node.child_by_field_name("declarator")
            if parameters_node:
                param_list_node = parameters_node.child_by_field_name("parameters")
                if param_list_node:
                    for param in param_list_node.children:
                        if param.type == "parameter_declaration":
                            type_node = param.child_by_field_name("type")
                            name_node = param.child_by_field_name("declarator")

                            param_type = (
                                ext.get_node_text(type_node)
                                if type_node
                                else None
                            )
                            param_name = (
                                ext.get_node_text(name_node)
                                if name_node
                                else None
                            )

                            args.append({"type": param_type, "name": param_name})

            # Get context info (function may be inside class)
            context_name, context_type, context_line = _cpp_get_parent_context(
                ext, node
            )
            class_context, _, _ = _cpp_get_parent_context(
                ext, node, types=("class_definition",)
            )

            call_data: dict[str, Any] = {
                "name": func_name,
                "full_name": full_name,
                "line_number": node.start_point[0] + 1,
                "args": args,
                "inferred_obj_type": None,
                "context": (context_name, context_type, context_line),
                "class_context": class_context,
                "lang": ext.language_name,
                "is_dependency": False,
            }
            calls.append(call_data)
    return calls


def find_cpp_variables(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find C++ variable declarations, excluding lambda assignments."""
    variables: list[dict[str, Any]] = []
    query_str = ext.config.queries["variables"]
    for match in ext.run_query(query_str, root_node):
        capture_name = match[1]
        node = match[0]

        if capture_name == "name":
            assignment_node = node.parent

            # Skip lambda assignments, they are handled by find_cpp_lambda_assignments
            right_node = assignment_node.child_by_field_name("value")
            if right_node and right_node.type == "lambda_expression":
                continue

            name = ext.get_node_text(node)
            value = ext.get_node_text(right_node) if right_node else None

            type_node = assignment_node.child_by_field_name("type")
            type_text = ext.get_node_text(type_node) if type_node else None

            context, _, _ = _cpp_get_parent_context(ext, node)
            class_context, _, _ = _cpp_get_parent_context(
                ext, node, types=("class_definition",)
            )

            variable_data: dict[str, Any] = {
                "name": name,
                "line_number": node.start_point[0] + 1,
                "value": value,
                "type": type_text,
                "context": context,
                "class_context": class_context,
                "lang": ext.language_name,
                "is_dependency": False,
            }
            variables.append(variable_data)
    return variables


# ===========================================================================
# C++  -- pre-scan
# ===========================================================================


def pre_scan_cpp(files: list[Path], parser_wrapper: Any) -> dict[str, list[str]]:
    """Quickly scans C++ files to build a map of top-level class, struct, and
    function names to their file paths.
    """
    imports_map: dict[str, list[str]] = {}

    query_str = """
        (class_specifier name: (type_identifier) @name)
        (struct_specifier name: (type_identifier) @name)
        (function_definition declarator: (function_declarator declarator: (identifier) @name))
    """

    for path in files:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                source_bytes = f.read().encode("utf-8")
                tree = parser_wrapper.parser.parse(source_bytes)

            for node, capture_name in execute_query(
                parser_wrapper.language, query_str, tree.root_node
            ):
                if capture_name == "name":
                    name = node.text.decode("utf-8")
                    imports_map.setdefault(name, []).append(str(path.resolve()))
        except Exception as e:
            warning_logger(f"Tree-sitter pre-scan failed for {path}: {e}")

    return imports_map


# ===========================================================================
# C#  -- helper functions
# ===========================================================================


def _csharp_get_parent_context(
    ext: GenericExtractor,
    node: Any,
    types: tuple[str, ...] = (
        "class_declaration",
        "struct_declaration",
        "function_declaration",
        "method_declaration",
    ),
) -> tuple[str | None, str | None, int | None]:
    """Find parent context for C# constructs."""
    curr = node.parent
    while curr:
        if curr.type in types:
            if curr.type in ("method_declaration", "function_declaration"):
                name_node = curr.child_by_field_name("name")
                return (
                    ext.get_node_text(name_node) if name_node else None,
                    curr.type,
                    curr.start_point[0] + 1,
                )
            else:
                # Classes, structs, etc.
                name_node = curr.child_by_field_name("name")
                return (
                    ext.get_node_text(name_node) if name_node else None,
                    curr.type,
                    curr.start_point[0] + 1,
                )
        curr = curr.parent
    return None, None, None


def _csharp_extract_parameters(ext: GenericExtractor, params_node: Any) -> list[str]:
    """Extract parameter names from a C# parameter list node."""
    params: list[str] = []
    if not params_node:
        return params

    # Iterate over parameter nodes in the parameter list
    for child in params_node.children:
        if child.type == "parameter":
            # find the identifier
            name_node = child.child_by_field_name("name")
            if name_node:
                params.append(ext.get_node_text(name_node))
            else:
                # Fallback: scan children for identifier if field name not present in this grammar version
                for sub in child.children:
                    if sub.type == "identifier":
                        params.append(ext.get_node_text(sub))
                        break
    return params


def _csharp_find_containing_type(
    ext: GenericExtractor, node: Any, source_code: str
) -> str | None:
    """Find the containing class, struct, interface, or record for a given node."""
    current = node.parent
    while current:
        if current.type in [
            "class_declaration",
            "struct_declaration",
            "interface_declaration",
            "record_declaration",
        ]:
            # Find the name of this type
            for child in current.children:
                if child.type == "identifier":
                    return source_code[child.start_byte : child.end_byte]
        current = current.parent
    return None


# ===========================================================================
# C#  -- finder hooks
# ===========================================================================


def find_csharp_functions(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find C# method, constructor, and local function declarations."""
    functions: list[dict[str, Any]] = []
    query_str = ext.config.queries["functions"]
    captures = list(ext.run_query(query_str, root_node))

    # We need access to source_code for byte-range slicing
    # Read it from the root_node
    source_code = root_node.text.decode("utf-8")

    for node, capture_name in captures:
        if capture_name == "function_node":
            try:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                name_captures = [
                    (n, cn)
                    for n, cn in captures
                    if cn == "name" and n.parent.id == node.id
                ]

                if name_captures:
                    name_node = name_captures[0][0]
                    func_name = source_code[name_node.start_byte : name_node.end_byte]

                    params_captures = [
                        (n, cn)
                        for n, cn in captures
                        if cn == "params" and n.parent.id == node.id
                    ]

                    parameters: list[str] = []
                    if params_captures:
                        params_node = params_captures[0][0]
                        parameters = _csharp_extract_parameters(ext, params_node)

                    # Extract attributes applied to this function
                    attributes: list[str] = []
                    if node.parent and node.parent.type == "attribute_list":
                        attr_text = source_code[
                            node.parent.start_byte : node.parent.end_byte
                        ]
                        attributes.append(attr_text)

                    # Find containing class/struct/interface
                    class_context = _csharp_find_containing_type(
                        ext, node, source_code
                    )

                    source_text = source_code[node.start_byte : node.end_byte]

                    func_data: dict[str, Any] = {
                        "name": func_name,
                        "args": parameters,
                        "attributes": attributes,
                        "line_number": start_line,
                        "end_line": end_line,
                        "path": ext.current_path,
                        "lang": ext.language_name,
                    }

                    # Add class context if found
                    if class_context:
                        func_data["class_context"] = class_context

                    if ext.index_source:
                        func_data["source"] = source_text

                    functions.append(func_data)

            except Exception as e:
                error_logger(f"Error parsing function in {ext.current_path}: {e}")
                continue

    return functions


def find_csharp_classes(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find C# class declarations with inheritance info."""
    query_str = ext.config.queries["classes"]
    captures = list(ext.run_query(query_str, root_node))
    source_code = root_node.text.decode("utf-8")
    return _csharp_parse_type_declarations(ext, captures, source_code, "Class")


def find_csharp_interfaces(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find C# interface declarations with inheritance info."""
    query_str = ext.config.queries["interfaces"]
    captures = list(ext.run_query(query_str, root_node))
    source_code = root_node.text.decode("utf-8")
    return _csharp_parse_type_declarations(ext, captures, source_code, "Interface")


def find_csharp_structs(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find C# struct declarations with inheritance info."""
    query_str = ext.config.queries["structs"]
    captures = list(ext.run_query(query_str, root_node))
    source_code = root_node.text.decode("utf-8")
    return _csharp_parse_type_declarations(ext, captures, source_code, "Struct")


def find_csharp_enums(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find C# enum declarations."""
    query_str = ext.config.queries["enums"]
    captures = list(ext.run_query(query_str, root_node))
    source_code = root_node.text.decode("utf-8")
    return _csharp_parse_type_declarations(ext, captures, source_code, "Enum")


def find_csharp_records(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find C# record declarations with inheritance info."""
    query_str = ext.config.queries["records"]
    captures = list(ext.run_query(query_str, root_node))
    source_code = root_node.text.decode("utf-8")
    return _csharp_parse_type_declarations(ext, captures, source_code, "Record")


def _csharp_parse_type_declarations(
    ext: GenericExtractor,
    captures: list[tuple[Any, str]],
    source_code: str,
    type_label: str,
) -> list[dict[str, Any]]:
    """Parse class, interface, struct, enum, or record declarations with inheritance info."""
    types: list[dict[str, Any]] = []

    # Map capture names based on type
    capture_map = {
        "Class": "class",
        "Interface": "interface",
        "Struct": "struct",
        "Enum": "enum",
        "Record": "record",
    }
    expected_capture = capture_map.get(type_label, "class")

    for node, capture_name in captures:
        if capture_name == expected_capture:
            try:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                name_captures = [
                    (n, cn)
                    for n, cn in captures
                    if cn == "name" and n.parent.id == node.id
                ]

                if name_captures:
                    name_node = name_captures[0][0]
                    type_name = source_code[name_node.start_byte : name_node.end_byte]

                    # Extract base classes/interfaces
                    bases: list[str] = []
                    bases_captures = [
                        (n, cn)
                        for n, cn in captures
                        if cn == "bases" and n.parent.id == node.id
                    ]

                    if bases_captures:
                        bases_node = bases_captures[0][0]
                        bases_text = source_code[
                            bases_node.start_byte : bases_node.end_byte
                        ]
                        # Parse base list: ": BaseClass, IInterface1, IInterface2"
                        bases_text = bases_text.strip().lstrip(":").strip()
                        if bases_text:
                            bases = [b.strip() for b in bases_text.split(",")]

                    source_text = source_code[node.start_byte : node.end_byte]

                    type_data: dict[str, Any] = {
                        "name": type_name,
                        "line_number": start_line,
                        "end_line": end_line,
                        "path": ext.current_path,
                        "lang": ext.language_name,
                    }

                    # Add bases if found
                    if bases:
                        type_data["bases"] = bases

                    if ext.index_source:
                        type_data["source"] = source_text

                    types.append(type_data)

            except Exception as e:
                error_logger(
                    f"Error parsing {type_label} in {ext.current_path}: {e}"
                )
                continue

    return types


def find_csharp_properties(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Parse C# properties."""
    properties: list[dict[str, Any]] = []
    query_str = ext.config.queries["properties"]
    captures = list(ext.run_query(query_str, root_node))
    source_code = root_node.text.decode("utf-8")

    for node, capture_name in captures:
        if capture_name == "property":
            try:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                name_captures = [
                    (n, cn)
                    for n, cn in captures
                    if cn == "name" and n.parent == node
                ]

                if name_captures:
                    name_node = name_captures[0][0]
                    prop_name = source_code[
                        name_node.start_byte : name_node.end_byte
                    ]

                    # Get property type from node children
                    prop_type = None
                    for child in node.children:
                        if child.type in [
                            "predefined_type",
                            "identifier",
                            "generic_name",
                            "nullable_type",
                            "array_type",
                        ]:
                            prop_type = source_code[
                                child.start_byte : child.end_byte
                            ]
                            break

                    # Find containing class/struct
                    class_context = _csharp_find_containing_type(
                        ext, node, source_code
                    )

                    source_text = source_code[node.start_byte : node.end_byte]

                    prop_data: dict[str, Any] = {
                        "name": prop_name,
                        "type": prop_type,
                        "line_number": start_line,
                        "end_line": end_line,
                        "path": ext.current_path,
                        "lang": ext.language_name,
                    }

                    if class_context:
                        prop_data["class_context"] = class_context

                    if ext.index_source:
                        prop_data["source"] = source_text

                    properties.append(prop_data)

            except Exception as e:
                error_logger(
                    f"Error parsing property in {ext.current_path}: {e}"
                )
                continue

    return properties


def find_csharp_imports(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find C# using directives."""
    imports: list[dict[str, Any]] = []
    query_str = ext.config.queries["imports"]
    captures = list(ext.run_query(query_str, root_node))
    source_code = root_node.text.decode("utf-8")

    for node, capture_name in captures:
        if capture_name == "import":
            try:
                import_text = source_code[node.start_byte : node.end_byte]
                # Match: using System.Collections.Generic; or using static System.Math;
                import_match = re.search(
                    r"using\s+(?:static\s+)?([^;]+)", import_text
                )
                if import_match:
                    import_path = import_match.group(1).strip()

                    # Check for alias: using MyAlias = System.Collections.Generic.List<int>;
                    alias = None
                    if "=" in import_path:
                        parts = import_path.split("=")
                        alias = parts[0].strip()
                        import_path = parts[1].strip()

                    import_data: dict[str, Any] = {
                        "name": import_path,
                        "full_import_name": import_path,
                        "line_number": node.start_point[0] + 1,
                        "alias": alias,
                        "context": (None, None),
                        "lang": ext.language_name,
                        "is_dependency": False,
                    }
                    imports.append(import_data)
            except Exception as e:
                error_logger(f"Error parsing import: {e}")
                continue

    return imports


def find_csharp_calls(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find C# invocation and object creation expressions."""
    calls: list[dict[str, Any]] = []
    seen_calls: set[str] = set()
    query_str = ext.config.queries["calls"]
    captures = list(ext.run_query(query_str, root_node))
    source_code = root_node.text.decode("utf-8")

    for node, capture_name in captures:
        if capture_name == "name":
            try:
                call_name = source_code[node.start_byte : node.end_byte]
                line_number = node.start_point[0] + 1

                # Avoid duplicates
                call_key = f"{call_name}_{line_number}"
                if call_key in seen_calls:
                    continue
                seen_calls.add(call_key)

                # Get context
                context_name, context_type, context_line = (
                    _csharp_get_parent_context(ext, node)
                )
                class_context = (
                    context_name
                    if context_type and "class" in context_type
                    else None
                )

                call_data: dict[str, Any] = {
                    "name": call_name,
                    "full_name": call_name,
                    "line_number": line_number,
                    "args": [],
                    "inferred_obj_type": None,
                    "context": (context_name, context_type, context_line),
                    "class_context": class_context,
                    "lang": ext.language_name,
                    "is_dependency": False,
                }
                calls.append(call_data)
            except Exception as e:
                error_logger(f"Error parsing call: {e}")
                continue

    return calls


# ===========================================================================
# C#  -- pre-scan
# ===========================================================================


def pre_scan_csharp(files: list[Path], parser_wrapper: Any) -> dict[str, list[str]]:
    """Pre-scan C# files to build a name-to-files mapping.

    Uses regex rather than tree-sitter for speed.
    """
    name_to_files: dict[str, list[str]] = {}

    for path in files:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Match class declarations
            class_matches = re.finditer(
                r"\b(?:public\s+|private\s+|protected\s+|internal\s+)?(?:static\s+)?(?:abstract\s+)?(?:sealed\s+)?(?:partial\s+)?class\s+(\w+)",
                content,
            )
            for match in class_matches:
                class_name = match.group(1)
                if class_name not in name_to_files:
                    name_to_files[class_name] = []
                name_to_files[class_name].append(str(path))

            # Match interface declarations
            interface_matches = re.finditer(
                r"\b(?:public\s+|private\s+|protected\s+|internal\s+)?(?:partial\s+)?interface\s+(\w+)",
                content,
            )
            for match in interface_matches:
                interface_name = match.group(1)
                if interface_name not in name_to_files:
                    name_to_files[interface_name] = []
                name_to_files[interface_name].append(str(path))

            # Match struct declarations
            struct_matches = re.finditer(
                r"\b(?:public\s+|private\s+|protected\s+|internal\s+)?(?:readonly\s+)?(?:partial\s+)?struct\s+(\w+)",
                content,
            )
            for match in struct_matches:
                struct_name = match.group(1)
                if struct_name not in name_to_files:
                    name_to_files[struct_name] = []
                name_to_files[struct_name].append(str(path))

            # Match record declarations
            record_matches = re.finditer(
                r"\b(?:public\s+|private\s+|protected\s+|internal\s+)?(?:sealed\s+)?record\s+(?:class\s+)?(\w+)",
                content,
            )
            for match in record_matches:
                record_name = match.group(1)
                if record_name not in name_to_files:
                    name_to_files[record_name] = []
                name_to_files[record_name].append(str(path))

        except Exception as e:
            error_logger(f"Error pre-scanning C# file {path}: {e}")

    return name_to_files


# ===========================================================================
# PHP  -- helper functions
# ===========================================================================


def _php_get_parent_context(
    ext: GenericExtractor, node: Any
) -> tuple[str | None, str | None, int | None]:
    """Get parent context for PHP constructs."""
    curr = node.parent
    while curr:
        if curr.type in (
            "function_definition",
            "method_declaration",
            "class_declaration",
            "interface_declaration",
            "trait_declaration",
        ):
            name_node = curr.child_by_field_name("name")
            return (
                ext.get_node_text(name_node) if name_node else None,
                curr.type,
                curr.start_point[0] + 1,
            )
        curr = curr.parent
    return None, None, None


# ===========================================================================
# PHP  -- finder hooks
# ===========================================================================


def find_php_functions(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find PHP function and method declarations."""
    functions: list[dict[str, Any]] = []
    seen_nodes: set[tuple[int, int, str]] = set()
    query_str = ext.config.queries["functions"]
    captures = list(ext.run_query(query_str, root_node))

    for node, capture_name in captures:
        if capture_name == "function_node":
            node_id = (node.start_byte, node.end_byte, node.type)
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)

            try:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                name_node = node.child_by_field_name("name")
                if name_node:
                    func_name = ext.get_node_text(name_node)

                    params_node = node.child_by_field_name("parameters")
                    parameters: list[str] = []
                    if params_node:
                        # PHP parameters: function($a, $b)
                        for child in params_node.children:
                            if (
                                "variable_name" in child.type
                                or "simple_parameter" in child.type
                            ):
                                # Extract variable name from simple_parameter
                                var_node = (
                                    child
                                    if "variable_name" in child.type
                                    else child.child_by_field_name("name")
                                )
                                if var_node:
                                    parameters.append(ext.get_node_text(var_node))

                    source_text = ext.get_node_text(node)

                    # Get class context
                    context_name, context_type, context_line = (
                        _php_get_parent_context(ext, node)
                    )

                    func_data: dict[str, Any] = {
                        "name": func_name,
                        "parameters": parameters,
                        "line_number": start_line,
                        "end_line": end_line,
                        "path": ext.current_path,
                        "lang": ext.language_name,
                        "context": context_name,
                        "context_type": context_type,
                        "class_context": context_name
                        if context_type
                        and (
                            "class" in context_type
                            or "interface" in context_type
                            or "trait" in context_type
                        )
                        else None,
                    }

                    if ext.index_source:
                        func_data["source"] = source_text

                    functions.append(func_data)

            except Exception as e:
                error_logger(f"Error parsing function in {ext.current_path}: {e}")
                continue

    return functions


def find_php_classes(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find PHP class, interface, and trait declarations.

    Returns only classes; interfaces and traits are returned by the separate
    ``find_php_interfaces`` and ``find_php_traits`` extras.
    """
    classes, _interfaces, _traits = _php_parse_types(ext, root_node)
    return classes


def find_php_interfaces(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find PHP interface declarations (extra finder)."""
    _classes, interfaces, _traits = _php_parse_types(ext, root_node)
    return interfaces


def find_php_traits(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find PHP trait declarations (extra finder)."""
    _classes, _interfaces, traits = _php_parse_types(ext, root_node)
    return traits


def _php_parse_types(
    ext: GenericExtractor, root_node: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Shared implementation that separates PHP classes, interfaces, and traits."""
    classes: list[dict[str, Any]] = []
    interfaces: list[dict[str, Any]] = []
    traits: list[dict[str, Any]] = []
    seen_nodes: set[tuple[int, int, str]] = set()

    query_str = ext.config.queries["classes"]
    captures = list(ext.run_query(query_str, root_node))

    for node, capture_name in captures:
        if capture_name in ("class", "interface", "trait"):
            node_id = (node.start_byte, node.end_byte, node.type)
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)

            try:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                name_node = node.child_by_field_name("name")
                if name_node:
                    type_name = ext.get_node_text(name_node)
                    source_text = ext.get_node_text(node)

                    bases: list[str] = []
                    # Look for extends/implements
                    base_clause_node = node.child_by_field_name(
                        "base_clause"
                    )  # extends
                    interfaces_clause_node = node.child_by_field_name(
                        "interfaces_clause"
                    )  # implements

                    if base_clause_node:
                        # Children of base_clause contain identifiers
                        for child in base_clause_node.children:
                            if child.type in ("name", "qualified_name"):
                                bases.append(ext.get_node_text(child))

                    if interfaces_clause_node:
                        for child in interfaces_clause_node.children:
                            if child.type in ("name", "qualified_name"):
                                bases.append(ext.get_node_text(child))

                    type_data: dict[str, Any] = {
                        "name": type_name,
                        "line_number": start_line,
                        "end_line": end_line,
                        "bases": bases,
                        "path": ext.current_path,
                        "lang": ext.language_name,
                    }
                    if ext.index_source:
                        type_data["source"] = source_text

                    if capture_name == "class":
                        classes.append(type_data)
                    elif capture_name == "interface":
                        interfaces.append(type_data)
                    elif capture_name == "trait":
                        traits.append(type_data)

            except Exception as e:
                error_logger(f"Error parsing type in {ext.current_path}: {e}")
                continue

    return classes, interfaces, traits


def find_php_imports(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find PHP use declarations."""
    imports: list[dict[str, Any]] = []
    query_str = ext.config.queries["imports"]
    captures = list(ext.run_query(query_str, root_node))

    for node, capture_name in captures:
        if capture_name == "import":
            try:
                import_text = ext.get_node_text(node)
                # use Foo\Bar as Baz;
                # Node usually has children: name (qualified_name), optional alias
                # Regex might be safer given tree complexity for `use`
                import_match = re.search(
                    r"use\s+([\w\\]+)(?:\s+as\s+(\w+))?", import_text
                )
                if import_match:
                    import_path = import_match.group(1).strip()
                    alias = (
                        import_match.group(2).strip()
                        if import_match.group(2)
                        else None
                    )

                    import_data: dict[str, Any] = {
                        "name": import_path,
                        "full_import_name": import_text,
                        "line_number": node.start_point[0] + 1,
                        "alias": alias,
                        "context": (None, None),
                        "lang": ext.language_name,
                        "is_dependency": False,
                    }
                    imports.append(import_data)
            except Exception as e:
                error_logger(f"Error parsing import: {e}")
                continue

    return imports


def find_php_calls(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find PHP function calls, method calls, scoped calls, and object creation."""
    calls: list[dict[str, Any]] = []
    seen_calls: set[str] = set()
    query_str = ext.config.queries["calls"]
    captures = list(ext.run_query(query_str, root_node))

    for node, capture_name in captures:
        # Handle 'name' capture which gives us the function name
        if capture_name == "name":
            try:
                call_name = ext.get_node_text(node)
                line_number = node.start_point[0] + 1

                # Ensure we identify the full call node
                call_node = node.parent
                while call_node and call_node.type not in (
                    "function_call_expression",
                    "member_call_expression",
                    "scoped_call_expression",
                ):
                    call_node = call_node.parent

                if not call_node:
                    continue  # It might be a name inside object creation or something we handle otherwise

                # Avoid duplicates
                call_key = f"{call_name}_{line_number}"
                if call_key in seen_calls:
                    continue
                seen_calls.add(call_key)

                # Extract arguments
                args: list[str] = []
                args_node = call_node.child_by_field_name("arguments")
                if args_node:
                    for arg in args_node.children:
                        if arg.type not in ("(", ")", ","):
                            args.append(ext.get_node_text(arg))

                full_name = call_name  # Default
                if call_node.type == "member_call_expression":
                    # $obj->method()
                    obj_node = call_node.child_by_field_name("object")
                    if obj_node:
                        receiver = ext.get_node_text(obj_node)
                        # Normalize -> to . for graph builder compatibility
                        full_name = f"{receiver}.{call_name}"
                elif call_node.type == "scoped_call_expression":
                    # Class::method()
                    scope_node = call_node.child_by_field_name("scope")
                    if scope_node:
                        receiver = ext.get_node_text(scope_node)
                        # Normalize :: to . for graph builder compatibility
                        full_name = f"{receiver}.{call_name}"

                ctx_name, ctx_type, ctx_line = _php_get_parent_context(ext, node)

                call_data: dict[str, Any] = {
                    "name": call_name,
                    "full_name": full_name,
                    "line_number": line_number,
                    "args": args,
                    "inferred_obj_type": None,
                    "context": (ctx_name, ctx_type, ctx_line),
                    "class_context": (ctx_name, ctx_line)
                    if ctx_type
                    and (
                        "class" in ctx_type
                        or "interface" in ctx_type
                        or "trait" in ctx_type
                    )
                    else (None, None),
                    "lang": ext.language_name,
                    "is_dependency": False,
                }
                calls.append(call_data)
            except Exception as e:
                error_logger(f"Error parsing call: {e}")
                continue

        # Handle object creation separately as capture is on the whole node
        elif capture_name == "call_node" and node.type == "object_creation_expression":
            try:
                line_number = node.start_point[0] + 1

                # Find class name (child not named 'arguments')
                class_name = "Unknown"
                for child in node.children:
                    if child.type in ("name", "qualified_name"):
                        class_name = ext.get_node_text(child)
                        break
                    if child.type == "variable_name":  # dynamic new $class()
                        class_name = ext.get_node_text(child)
                        break

                call_key = f"new {class_name}_{line_number}"
                if call_key in seen_calls:
                    continue
                seen_calls.add(call_key)

                args = []
                args_node = node.child_by_field_name("arguments")
                if args_node:
                    for arg in args_node.children:
                        if arg.type not in ("(", ")", ","):
                            args.append(ext.get_node_text(arg))

                full_name = class_name  # For GraphBuilder to link to Class

                ctx_name, ctx_type, ctx_line = _php_get_parent_context(ext, node)

                call_data = {
                    "name": class_name,
                    "full_name": full_name,  # Usually we want the class name here so GB can link to Class node
                    "line_number": line_number,
                    "args": args,
                    "inferred_obj_type": None,
                    "context": (ctx_name, ctx_type, ctx_line),
                    "class_context": (ctx_name, ctx_line)
                    if ctx_type
                    and (
                        "class" in ctx_type
                        or "interface" in ctx_type
                        or "trait" in ctx_type
                    )
                    else (None, None),
                    "lang": ext.language_name,
                    "is_dependency": False,
                }
                calls.append(call_data)
            except Exception:
                continue

    return calls


def find_php_variables(
    ext: GenericExtractor, root_node: Any
) -> list[dict[str, Any]]:
    """Find PHP variable references with type inference from assignments."""
    variables: list[dict[str, Any]] = []
    seen_vars: set[int] = set()
    query_str = ext.config.queries["variables"]
    captures = list(ext.run_query(query_str, root_node))

    for node, capture_name in captures:
        if capture_name == "variable":
            try:
                var_name = ext.get_node_text(node)
                start_line = node.start_point[0] + 1

                start_byte = node.start_byte
                if start_byte in seen_vars:
                    continue
                seen_vars.add(start_byte)

                ctx_name, ctx_type, ctx_line = _php_get_parent_context(ext, node)

                # Infer type from assignment
                inferred_type = "mixed"
                parent = node.parent
                if parent and parent.type == "assignment_expression":
                    # $var = new Class();
                    left = parent.child_by_field_name("left")
                    right = parent.child_by_field_name("right")

                    # Ensure we are looking at the left side variable
                    if (
                        left == node
                        and right
                        and right.type == "object_creation_expression"
                    ):
                        # Extract class name from right side
                        for child in right.children:
                            if child.type in ("name", "qualified_name"):
                                inferred_type = ext.get_node_text(child)
                                break

                variables.append(
                    {
                        "name": var_name,
                        "type": inferred_type,
                        "line_number": start_line,
                        "path": ext.current_path,
                        "lang": ext.language_name,
                        "context": ctx_name,
                        "class_context": ctx_name
                        if ctx_type
                        and (
                            "class" in ctx_type
                            or "interface" in ctx_type
                            or "trait" in ctx_type
                        )
                        else None,
                    }
                )
            except Exception:
                continue

    return variables


# ===========================================================================
# PHP  -- pre-scan
# ===========================================================================


def pre_scan_php(files: list[Path], parser_wrapper: Any) -> dict[str, list[str]]:
    """Pre-scan PHP files (currently a no-op placeholder)."""
    name_to_files: dict[str, list[str]] = {}
    return name_to_files


# ============================================================================
# SWIFT
# ============================================================================

# ---------------------------------------------------------------------------
# Swift helpers (module-level private)
# ---------------------------------------------------------------------------

def _swift_get_parent_context(
    ext: GenericExtractor, node: Any
) -> tuple[str | None, str | None, int | None]:
    """Walk up AST using Swift-specific context rules."""
    curr = node.parent
    while curr:
        if curr.type == "function_declaration":
            name_node = None
            for child in curr.children:
                if child.type == "simple_identifier":
                    name_node = child
                    break
            return (
                ext.get_node_text(name_node) if name_node else None,
                curr.type,
                curr.start_point[0] + 1,
            )
        if curr.type in ("class_declaration", "struct_declaration", "enum_declaration", "protocol_declaration"):
            for child in curr.children:
                if child.type == "type_identifier":
                    return (
                        ext.get_node_text(child),
                        curr.type,
                        curr.start_point[0] + 1,
                    )
        if curr.type == "init_declaration":
            # For initializers, return the parent class/struct name
            parent = curr.parent
            if parent and parent.type in ("class_body", "struct_body"):
                grandparent = parent.parent
                if grandparent:
                    for child in grandparent.children:
                        if child.type == "type_identifier":
                            return (
                                ext.get_node_text(child),
                                grandparent.type,
                                grandparent.start_point[0] + 1,
                            )
            return ("init", curr.type, curr.start_point[0] + 1)
        curr = curr.parent
    return None, None, None


def _swift_extract_parameter_name(ext: GenericExtractor, param_node: Any) -> str | None:
    """Extract parameter name from a parameter node."""
    # Swift parameters can have external and internal names
    # parameter: external_name? internal_name: type
    for child in param_node.children:
        if child.type == "simple_identifier":
            return ext.get_node_text(child)
    return None


# ---------------------------------------------------------------------------
# Swift finder hooks
# ---------------------------------------------------------------------------

def find_swift_functions(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Swift functions and init declarations."""
    query_str = ext.config.queries.get("functions")
    if not query_str:
        return []

    functions: list[dict[str, Any]] = []
    seen_nodes: set[tuple[int, int, str]] = set()

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name in ("function_node", "init_node"):
            node_id = (node.start_byte, node.end_byte, node.type)
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)

            try:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                # Get function name
                func_name = "init" if capture_name == "init_node" else None
                if capture_name == "function_node":
                    for child in node.children:
                        if child.type == "simple_identifier":
                            func_name = ext.get_node_text(child)
                            break

                if not func_name:
                    continue

                # Extract parameters
                parameters: list[str] = []
                for child in node.children:
                    if child.type == "parameter":
                        param_name = _swift_extract_parameter_name(ext, child)
                        if param_name:
                            parameters.append(param_name)

                source_text = ext.get_node_text(node)
                context_name, context_type, context_line = _swift_get_parent_context(ext, node)

                func_data: dict[str, Any] = {
                    "name": func_name,
                    "args": parameters,
                    "line_number": start_line,
                    "end_line": end_line,
                    "path": ext.current_path,
                    "lang": ext.language_name,
                    "context": context_name,
                    "class_context": context_name if context_type and ("class" in context_type or "struct" in context_type) else None,
                }

                if ext.index_source:
                    func_data["source"] = source_text

                functions.append(func_data)

            except Exception as e:
                error_logger(f"Error parsing function in {ext.current_path}: {e}")
                continue

    return functions


def find_swift_classes(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Swift classes (only the class_declaration nodes)."""
    query_str = ext.config.queries.get("classes")
    if not query_str:
        return []

    classes, _structs, _enums, _protocols = _swift_parse_all_types(ext, root_node, query_str)
    return classes


def find_swift_structs(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Swift struct declarations (extra finder)."""
    query_str = ext.config.queries.get("classes")
    if not query_str:
        return []

    _classes, structs, _enums, _protocols = _swift_parse_all_types(ext, root_node, query_str)
    return structs


def find_swift_enums(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Swift enum declarations (extra finder)."""
    query_str = ext.config.queries.get("classes")
    if not query_str:
        return []

    _classes, _structs, enums, _protocols = _swift_parse_all_types(ext, root_node, query_str)
    return enums


def find_swift_protocols(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Swift protocol declarations (extra finder)."""
    query_str = ext.config.queries.get("classes")
    if not query_str:
        return []

    _classes, _structs, _enums, protocols = _swift_parse_all_types(ext, root_node, query_str)
    return protocols


def _swift_parse_all_types(
    ext: GenericExtractor, root_node: Any, query_str: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse all four Swift type declarations from a single query, returning (classes, structs, enums, protocols)."""
    classes: list[dict[str, Any]] = []
    structs: list[dict[str, Any]] = []
    enums: list[dict[str, Any]] = []
    protocols: list[dict[str, Any]] = []
    seen_nodes: set[tuple[int, int, str]] = set()

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name in ("class", "struct", "enum", "protocol"):
            node_id = (node.start_byte, node.end_byte, node.type)
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)

            try:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                # Find name
                type_name = "Anonymous"
                for child in node.children:
                    if child.type == "type_identifier":
                        type_name = ext.get_node_text(child)
                        break

                source_text = ext.get_node_text(node)

                # Extract inheritance/protocol conformance
                bases: list[str] = []
                for child in node.children:
                    if child.type == "type_inheritance_clause":
                        for subchild in child.children:
                            if subchild.type == "type_identifier":
                                bases.append(ext.get_node_text(subchild))

                type_data: dict[str, Any] = {
                    "name": type_name,
                    "line_number": start_line,
                    "end_line": end_line,
                    "bases": bases,
                    "path": ext.current_path,
                    "lang": ext.language_name,
                }

                if ext.index_source:
                    type_data["source"] = source_text

                if capture_name == "class":
                    classes.append(type_data)
                elif capture_name == "struct":
                    structs.append(type_data)
                elif capture_name == "enum":
                    enums.append(type_data)
                elif capture_name == "protocol":
                    protocols.append(type_data)

            except Exception as e:
                error_logger(f"Error parsing type in {ext.current_path}: {e}")
                continue

    return classes, structs, enums, protocols


def find_swift_imports(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Swift import declarations."""
    query_str = ext.config.queries.get("imports")
    if not query_str:
        return []

    imports: list[dict[str, Any]] = []

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "import":
            try:
                text = ext.get_node_text(node)
                # import Foundation
                # import UIKit
                parts = text.replace('import ', '').strip().split()
                module_name = parts[0] if parts else ""

                if module_name:
                    imports.append({
                        "name": module_name,
                        "full_import_name": module_name,
                        "line_number": node.start_point[0] + 1,
                        "alias": None,
                        "context": (None, None),
                        "lang": ext.language_name,
                        "is_dependency": False,
                    })
            except Exception:
                continue

    return imports


def find_swift_calls(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Swift call expressions with type inference."""
    query_str = ext.config.queries.get("calls")
    if not query_str:
        return []

    calls: list[dict[str, Any]] = []
    seen_calls: set[str] = set()

    # Index variables for fast lookup
    var_map: dict[tuple[str, str | None], str] = {}
    for v in ext._parsed_variables:
        key = (v['name'], v['context'])
        var_map[key] = v['type']

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "call_node":
            try:
                start_line = node.start_point[0] + 1

                call_name = "unknown"
                base_obj = None

                # Extract function name from call expression
                # call_expression can have various structures
                first_child = node.children[0] if node.children else None

                if first_child:
                    if first_child.type == "simple_identifier":
                        call_name = ext.get_node_text(first_child)
                    elif first_child.type == "navigation_expression":
                        # obj.method() pattern
                        for child in first_child.children:
                            if child.type == "simple_identifier":
                                if not base_obj:
                                    base_obj = ext.get_node_text(child)
                                else:
                                    call_name = ext.get_node_text(child)

                if call_name == "unknown":
                    continue

                full_name = f"{base_obj}.{call_name}" if base_obj else call_name
                ctx_name, ctx_type, ctx_line = _swift_get_parent_context(ext, node)

                # Type inference
                inferred_type = None
                if base_obj:
                    inferred_type = var_map.get((base_obj, ctx_name))
                    if not inferred_type:
                        inferred_type = var_map.get((base_obj, None))
                    if not inferred_type:
                        for (vname, vctx), vtype in var_map.items():
                            if vname == base_obj:
                                inferred_type = vtype
                                break

                calls.append({
                    "name": call_name,
                    "full_name": full_name,
                    "line_number": start_line,
                    "args": [],
                    "inferred_obj_type": inferred_type,
                    "context": [None, ctx_type, ctx_line],
                    "class_context": [None, None],
                    "lang": ext.language_name,
                    "is_dependency": False,
                })
            except Exception:
                continue

    return calls


def find_swift_variables(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Swift property and constant declarations."""
    query_str = ext.config.queries.get("variables")
    if not query_str:
        return []

    variables: list[dict[str, Any]] = []
    seen_vars: set[tuple[str, int]] = set()

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name in ("variable", "constant", "pattern"):
            try:
                start_line = node.start_point[0] + 1
                ctx_name, ctx_type, ctx_line = _swift_get_parent_context(ext, node)

                var_name = "unknown"
                var_type = "Unknown"

                # Try to extract variable name
                if capture_name == "pattern":
                    var_name = ext.get_node_text(node)
                else:
                    for child in node.children:
                        if child.type == "simple_identifier":
                            var_name = ext.get_node_text(child)
                            break
                        elif child.type == "pattern_binding":
                            for subchild in child.children:
                                if subchild.type == "simple_identifier":
                                    var_name = ext.get_node_text(subchild)
                                    break

                # Try to extract type annotation
                for child in node.children:
                    if child.type == "type_annotation":
                        for subchild in child.children:
                            if subchild.type == "type_identifier":
                                var_type = ext.get_node_text(subchild)
                                break

                if var_name != "unknown":
                    var_key = (var_name, start_line)
                    if var_key not in seen_vars:
                        seen_vars.add(var_key)
                        variables.append({
                            "name": var_name,
                            "type": var_type,
                            "line_number": start_line,
                            "path": ext.current_path,
                            "lang": ext.language_name,
                            "context": ctx_name,
                            "class_context": ctx_name if ctx_type and ("class" in ctx_type or "struct" in ctx_type) else None,
                        })
            except Exception:
                continue

    return variables


# ---------------------------------------------------------------------------
# Swift pre-scan
# ---------------------------------------------------------------------------

def pre_scan_swift(files: list[Path], parser_wrapper: Any) -> dict[str, list[str]]:
    """Pre-scan Swift files to build a map of class/struct/enum/protocol names to file paths."""
    name_to_files: dict[str, list[str]] = {}
    for path in files:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Extract classes, structs, enums, protocols
            matches = re.finditer(r'\b(class|struct|enum|protocol)\s+(\w+)', content)

            for match in matches:
                name = match.group(2)
                if name not in name_to_files:
                    name_to_files[name] = []
                name_to_files[name].append(str(path))

        except Exception:
            pass
    return name_to_files


# ============================================================================
# KOTLIN
# ============================================================================

# ---------------------------------------------------------------------------
# Kotlin helpers (module-level private)
# ---------------------------------------------------------------------------

def _kotlin_get_parent_context(
    ext: GenericExtractor, node: Any
) -> tuple[str | None, str | None, int | None]:
    """Walk up AST using Kotlin-specific context rules."""
    curr = node.parent
    while curr:
        if curr.type in ("function_declaration",):
            name_node = None
            for child in curr.children:
                if child.type == "simple_identifier":
                    name_node = child
                    break
            return (
                ext.get_node_text(name_node) if name_node else None,
                curr.type,
                curr.start_point[0] + 1,
            )
        if curr.type in ("class_declaration", "object_declaration"):
            for child in curr.children:
                if child.type in ("simple_identifier", "type_identifier"):
                    return (
                        ext.get_node_text(child),
                        curr.type,
                        curr.start_point[0] + 1,
                    )
            # Check for secondary constructors
            if curr.type == "secondary_constructor":
                return (
                    "constructor",
                    curr.type,
                    curr.start_point[0] + 1,
                )

        if curr.type == "companion_object":
            name = "Companion"
            for child in curr.children:
                if child.type in ("simple_identifier", "type_identifier"):
                    name = ext.get_node_text(child)
                    break
            return (
                name,
                curr.type,
                curr.start_point[0] + 1,
            )

        # Handle anonymous objects (object_literal)
        if curr.type == "object_literal":
            # checking if it is assigned to a variable to get a name?
            # or simply "AnonymousObject"
            # It's usually hard to name them without variable context.
            # We can check if parent is property/variable declaration
            name = "AnonymousObject"
            return (
                name,
                curr.type,
                curr.start_point[0] + 1,
            )

        curr = curr.parent
    return None, None, None


def _kotlin_extract_parameter_names(params_text: str) -> list[str]:
    """
    Extracts parameter names from a Kotlin parameter list string.
    Handles nested generics like Map<String, Int>.

    Args:
        params_text: The text content of function_value_parameters node,
                     e.g. "(a: Int, b: Map<String, Int>)"

    Returns:
        List of parameter names.
    """
    params: list[str] = []
    if not params_text:
        return params

    # Remove outer parentheses
    clean = params_text.strip()
    if clean.startswith('(') and clean.endswith(')'):
        clean = clean[1:-1]

    if not clean.strip():
        return params

    # Robust splitting by comma, respecting brackets <>, (), [], {}
    current_param: list[str] = []
    depth_angle = 0  # < >
    depth_round = 0  # ( )
    depth_square = 0  # [ ]
    depth_curly = 0  # { }

    raw_params: list[str] = []

    for char in clean:
        if char == '<':
            depth_angle += 1
        elif char == '>':
            depth_angle -= 1
        elif char == '(':
            depth_round += 1
        elif char == ')':
            depth_round -= 1
        elif char == '[':
            depth_square += 1
        elif char == ']':
            depth_square -= 1
        elif char == '{':
            depth_curly += 1
        elif char == '}':
            depth_curly -= 1

        if char == ',' and depth_angle == 0 and depth_round == 0 and depth_square == 0 and depth_curly == 0:
            raw_params.append("".join(current_param).strip())
            current_param = []
        else:
            current_param.append(char)

    if current_param:
        raw_params.append("".join(current_param).strip())

    # Process each raw parameter string to extract name
    # Format: "val x: Int", "override var y: String", "@Ann z: Int", "a: Int = 5"
    for p in raw_params:
        if not p:
            continue

        # Split by ':' to separate name/modifiers from Type
        # Using the first ':' usually works, assuming name doesn't contain ':'
        colon_index = p.find(':')
        if colon_index != -1:
            lhs = p[:colon_index].strip()
        else:
            # Could be a parameter without type? (not common in Kotlin unless lambda destructuring)
            # Or "var x = 5" (unlikely in func params)
            # Just take the whole string if no colon?
            lhs = p.strip()

        # LHS contains keywords (val, var), annotations (@Foo), modifiers (crossinline, noinline, vararg)
        # and the parameter name. The parameter name is usually the LAST identifier.

        if not lhs:
            continue

        tokens = lhs.split()
        if tokens:
            # The name is the last token
            params.append(tokens[-1])

    return params


# ---------------------------------------------------------------------------
# Kotlin finder hooks
# ---------------------------------------------------------------------------

def find_kotlin_functions(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Kotlin function declarations."""
    query_str = ext.config.queries.get("functions")
    if not query_str:
        return []

    functions: list[dict[str, Any]] = []
    seen_nodes: set[tuple[int, int, str]] = set()

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "function_node":
            node_id = (node.start_byte, node.end_byte, node.type)
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)

            try:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                # Manual child lookup
                name_node = None
                for child in node.children:
                    if child.type == "simple_identifier":
                        name_node = child
                        break

                if name_node:
                    func_name = ext.get_node_text(name_node)

                    params_node = None
                    for child in node.children:
                        if child.type == "function_value_parameters":
                            params_node = child
                            break

                    parameters: list[str] = []
                    if params_node:
                        params_text = ext.get_node_text(params_node)
                        parameters = _kotlin_extract_parameter_names(params_text)

                    source_text = ext.get_node_text(node)

                    context_name, context_type, context_line = _kotlin_get_parent_context(ext, node)

                    func_data: dict[str, Any] = {
                        "name": func_name,
                        "args": parameters,
                        "line_number": start_line,
                        "end_line": end_line,
                        "path": ext.current_path,
                        "lang": ext.language_name,
                        "context": context_name,
                        "class_context": context_name if context_type and ("class" in context_type or "object" in context_type) else None,
                    }

                    if ext.index_source:
                        func_data["source"] = source_text

                    functions.append(func_data)

            except Exception as e:
                error_logger(f"Error parsing function in {ext.current_path}: {e}")
                continue

    return functions


def find_kotlin_classes(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Kotlin class/object/companion_object declarations."""
    query_str = ext.config.queries.get("classes")
    if not query_str:
        return []

    classes: list[dict[str, Any]] = []
    seen_nodes: set[tuple[int, int, str]] = set()

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "class":
            node_id = (node.start_byte, node.end_byte, node.type)
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)

            try:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                # Find name child (type_identifier or simple_identifier)
                class_name = "Anonymous"
                if node.type == "companion_object":
                    class_name = "Companion"  # Default name

                for child in node.children:
                    if child.type in ("type_identifier", "simple_identifier"):
                        class_name = ext.get_node_text(child)
                        break

                source_text = ext.get_node_text(node)

                bases: list[str] = []
                # Check for delegation specifiers
                # class_declaration -> delegation_specifier

                for child in node.children:
                    if child.type == "delegation_specifier":
                        # children: constructor_invocation or user_type
                        for specifier in child.children:
                            # constructor_invocation -> user_type -> type_identifier
                            # user_type -> type_identifier

                            # We want the text of the type
                            if specifier.type == "constructor_invocation":
                                # child 0 is typically user_type
                                for sub in specifier.children:
                                    if sub.type == "user_type":
                                        bases.append(ext.get_node_text(sub))
                                        break
                            elif specifier.type == "user_type":
                                bases.append(ext.get_node_text(specifier))
                            elif specifier.type == "explicit_delegation":
                                # Not handling simple yet, uses 'by'
                                pass

                class_data: dict[str, Any] = {
                    "name": class_name,
                    "line_number": start_line,
                    "end_line": end_line,
                    "bases": bases,
                    "path": ext.current_path,
                    "lang": ext.language_name,
                }

                if ext.index_source:
                    class_data["source"] = source_text

                classes.append(class_data)

            except Exception as e:
                error_logger(f"Error parsing class in {ext.current_path}: {e}")
                continue

    return classes


def find_kotlin_imports(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Kotlin import declarations."""
    query_str = ext.config.queries.get("imports")
    if not query_str:
        return []

    imports: list[dict[str, Any]] = []

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "import":
            try:
                # import_header -> "import" identifier (import_alias)?
                text = ext.get_node_text(node)
                # remove 'import '
                path = text.replace('import ', '').strip().split(' as ')[0].strip()
                alias = None
                if ' as ' in text:
                    alias = text.split(' as ')[1].strip()

                imports.append({
                    "name": path,
                    "full_import_name": path,
                    "line_number": node.start_point[0] + 1,
                    "alias": alias,
                    "context": (None, None),
                    "lang": ext.language_name,
                    "is_dependency": False,
                })
            except Exception:
                continue

    return imports


def find_kotlin_calls(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Kotlin call expressions with navigation_suffix handling and type inference."""
    query_str = ext.config.queries.get("calls")
    if not query_str:
        return []

    calls: list[dict[str, Any]] = []
    seen_calls: set[str] = set()

    # Index variables for fast lookup: (name, context) -> type
    var_map: dict[tuple[str, str | None], str] = {}
    for v in ext._parsed_variables:
        key = (v['name'], v['context'])
        var_map[key] = v['type']
        # Fallback for null context or partial match could be added
        # For class props: (name, class_context) might work if local lookup fails?

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "call_node":
            try:
                # navigation_expression check

                start_line = node.start_point[0] + 1

                call_name = "unknown"
                base_obj = None

                # call_expression usually has children:
                # simple_identifier (func name)
                # or navigation_expression (obj.method)

                # Heuristic for base object:
                # If navigation_expression -> child[0] is base, child[1] is suffix (method)

                # We need to look deeper into the call_expression structure.
                # call_expression -> (simple_identifier)
                # OR call_expression -> (navigation_expression (simple_identifier) (navigation_suffix (simple_identifier) ...))
                # OR call_expression -> (navigation_expression (call_expression) ...)  (chained)

                # Simplified traversal to find the "function name" and "receiver"

                # If it's a direct call: foo()
                # If it's a method call: x.foo()

                # Tree-sitter struct:
                # (call_expression (simple_identifier) (call_suffix ...))  -> name = simple_identifier
                # (call_expression (navigation_expression (simple_identifier) (navigation_suffix (simple_identifier))) (call_suffix))
                #  -> name = 2nd simple_identifier, base = 1st simple_identifier

                # Let's verify children
                children = node.children
                first_child = children[0]

                if first_child.type == "simple_identifier":
                    call_name = ext.get_node_text(first_child)
                    # No explicit base object
                elif first_child.type == "navigation_expression":
                    # x.foo
                    # children: operand (x), operator (.), suffix (foo)
                    # Usually 3 children?
                    # Let's inspect nav expression children
                    nav_children = first_child.children
                    if len(nav_children) >= 2:
                        # operand is 0
                        operand = nav_children[0]
                        # last one is suffix?
                        suffix = nav_children[-1]

                        # Suffix usually contains the method name in a navigation_suffix node or directly?
                        # Suffix is (navigation_suffix (simple_identifier)) usually.

                        if suffix.type == "navigation_suffix":
                            # (navigation_suffix (simple_identifier))
                            for c in suffix.children:
                                if c.type == "simple_identifier":
                                    call_name = ext.get_node_text(c)
                                    break
                        elif suffix.type == "simple_identifier":
                            call_name = ext.get_node_text(suffix)

                        # Base object
                        base_obj = ext.get_node_text(operand)

                if call_name == "unknown":
                    continue

                full_name = f"{base_obj}.{call_name}" if base_obj else call_name

                ctx_name, ctx_type, ctx_line = _kotlin_get_parent_context(ext, node)

                # Inference
                inferred_type = None
                if base_obj:
                    # Lookup base_obj in variables
                    # Try exact context
                    inferred_type = var_map.get((base_obj, ctx_name))
                    if not inferred_type:
                        # Try class context if we are in a method
                        # This logic is approximate.
                        # If we are in method 'foo' of 'ClassA', and 'base_obj' refers to a property of 'ClassA',
                        # var_map entry would have context 'ClassA'.
                        # But our 'variables' parsing puts context as 'ClassA' for props.
                        # But 'ctx_name' here is 'foo'.
                        # We need to know 'foo' is in 'ClassA'.
                        # 'get_parent_context' returns immediate parent.
                        pass
                        # Fallback: check global/file scope (context=None)
                        if not inferred_type:
                            inferred_type = var_map.get((base_obj, None))

                        # Fallback: check if any variable named base_obj exists (loose match)
                        if not inferred_type:
                            for (vname, vctx), vtype in var_map.items():
                                if vname == base_obj:
                                    inferred_type = vtype
                                    break

                calls.append({
                    "name": call_name,
                    "full_name": full_name,
                    "line_number": start_line,
                    "args": [],  # Simplified
                    "inferred_obj_type": inferred_type,
                    "context": [None, ctx_type, ctx_line],  # Keeping format compatible
                    "class_context": [None, None],
                    "lang": ext.language_name,
                    "is_dependency": False,
                })
            except Exception:
                continue

    return calls


def find_kotlin_variables(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Kotlin property declarations with type inference."""
    query_str = ext.config.queries.get("variables")
    if not query_str:
        return []

    variables: list[dict[str, Any]] = []
    seen_vars: set[tuple[str, int]] = set()

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "variable":
            try:
                start_line = node.start_point[0] + 1
                ctx_name, ctx_type, ctx_line = _kotlin_get_parent_context(ext, node)

                # Destructuring declaration
                if "destructuring" in node.type:
                    pass

                # Regular property/variable
                var_name = "unknown"
                var_type = "Unknown"

                var_decl = None
                for child in node.children:
                    if child.type == "variable_declaration":
                        var_decl = child
                        break

                if var_decl:
                    # Check for name and type in variable_declaration
                    for child in var_decl.children:
                        if child.type == "simple_identifier":
                            var_name = ext.get_node_text(child)

                        if child.type == "user_type":
                            var_type = ext.get_node_text(child)

                # Attempt inference from initializer if type is unknown
                if var_type == "Unknown":
                    # property_declaration -> expression (e.g. call_expression)
                    for child in node.children:
                        if child.type == "call_expression":
                            # call_expression -> simple_identifier (constructor)
                            for sub in child.children:
                                if sub.type == "simple_identifier":
                                    var_type = ext.get_node_text(sub)
                                    break
                            if var_type != "Unknown":
                                break

                if var_name != "unknown":
                    variables.append({
                        "name": var_name,
                        "type": var_type,
                        "line_number": start_line,
                        "path": ext.current_path,
                        "lang": ext.language_name,
                        "context": ctx_name,
                        "class_context": ctx_name if ctx_type and ("class" in ctx_type or "object" in ctx_type) else None,
                    })
            except Exception:
                continue

    return variables


# ---------------------------------------------------------------------------
# Kotlin pre-scan
# ---------------------------------------------------------------------------

def pre_scan_kotlin(files: list[Path], parser_wrapper: Any) -> dict[str, list[str]]:
    """Pre-scan Kotlin files to build a map of class/object/interface names to file paths."""
    name_to_files: dict[str, list[str]] = {}
    for path in files:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 1. Extract package
            # package com.example.project
            package_name = ""
            pkg_match = re.search(r'^\s*package\s+([\w\.]+)', content, re.MULTILINE)
            if pkg_match:
                package_name = pkg_match.group(1)

            # 2. Extract classes/objects/interfaces/typealiases
            matches = re.finditer(r'\b(class|interface|object|typealias)\s+(\w+)', content)

            for match in matches:
                name = match.group(2)
                # Map simple name
                if name not in name_to_files:
                    name_to_files[name] = []
                name_to_files[name].append(str(path))

                # If package exists, map FQN
                if package_name:
                    fqn = f"{package_name}.{name}"
                    if fqn not in name_to_files:
                        name_to_files[fqn] = []
                    name_to_files[fqn].append(str(path))

        except Exception:
            pass
    return name_to_files


# ============================================================================
# SCALA
# ============================================================================

# ---------------------------------------------------------------------------
# Scala helpers (module-level private)
# ---------------------------------------------------------------------------

def _scala_get_parent_context(
    ext: GenericExtractor, node: Any
) -> tuple[str | None, str | None, int | None]:
    """Walk up AST using Scala-specific context rules (uses child_by_field_name)."""
    curr = node.parent
    while curr:
        if curr.type == "function_definition":
            name_node = curr.child_by_field_name("name")
            return (
                ext.get_node_text(name_node) if name_node else None,
                curr.type,
                curr.start_point[0] + 1,
            )
        if curr.type in ("class_definition", "object_definition", "trait_definition"):
            name_node = curr.child_by_field_name("name")
            return (
                ext.get_node_text(name_node) if name_node else None,
                curr.type,
                curr.start_point[0] + 1,
            )
        curr = curr.parent
    return None, None, None


def _scala_extract_parameter_names(params_text: str) -> list[str]:
    """Simple extraction for Scala: (a: Int, b: String)."""
    params: list[str] = []
    if not params_text:
        return params
    clean = params_text.strip("()")
    if not clean:
        return params

    # Split by comma, respecting generics []
    # Scala generics use []

    # TODO: Reuse regex/parsing logic from other parsers or write simple one
    # For now, simplistic split
    parts = clean.split(',')
    for p in parts:
        # removing type: 'name: Type'
        if ':' in p:
            name = p.split(':')[0].strip()
            # Remove modifiers like 'implicit', 'override', etc.
            tokens = name.split()
            if tokens:
                params.append(tokens[-1])
        else:
            # maybe just name?
            params.append(p.strip())
    return params


# ---------------------------------------------------------------------------
# Scala finder hooks
# ---------------------------------------------------------------------------

def find_scala_functions(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Scala function definitions."""
    query_str = ext.config.queries.get("functions")
    if not query_str:
        return []

    functions: list[dict[str, Any]] = []
    seen_nodes: set[tuple[int, int, str]] = set()

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "function_node":
            node_id = (node.start_byte, node.end_byte, node.type)
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)

            try:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                name_node = node.child_by_field_name("name")
                if name_node:
                    func_name = ext.get_node_text(name_node)

                    params_node = node.child_by_field_name("parameters")
                    parameters: list[str] = []
                    if params_node:
                        params_text = ext.get_node_text(params_node)
                        parameters = _scala_extract_parameter_names(params_text)

                    source_text = ext.get_node_text(node)

                    context_name, context_type, context_line = _scala_get_parent_context(ext, node)

                    func_data: dict[str, Any] = {
                        "name": func_name,
                        "parameters": parameters,
                        "args": parameters,  # 'args' is sometimes used instead of 'parameters'
                        "line_number": start_line,
                        "end_line": end_line,
                        "path": ext.current_path,
                        "lang": ext.language_name,
                        "context": context_name,
                        "class_context": context_name if context_type and "class" in str(context_type) or "object" in str(context_type) or "trait" in str(context_type) else None,
                    }

                    if ext.index_source:
                        func_data["source"] = source_text

                    functions.append(func_data)

            except Exception as e:
                error_logger(f"Error parsing function in {ext.current_path}: {e}")
                continue

    return functions


def find_scala_classes(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Scala class and object definitions (excludes traits)."""
    query_str = ext.config.queries.get("classes")
    if not query_str:
        return []

    all_classes = _scala_parse_all_classes(ext, root_node, query_str)

    final_classes: list[dict[str, Any]] = []

    for item in all_classes:
        item_type = item.get('type', 'class')
        if item_type == 'trait':
            pass  # excluded from classes, goes to traits finder
        elif item_type == 'object':
            item['is_object'] = True
            final_classes.append(item)
        else:
            final_classes.append(item)

    return final_classes


def find_scala_traits(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Scala trait definitions (extra finder)."""
    query_str = ext.config.queries.get("classes")
    if not query_str:
        return []

    all_classes = _scala_parse_all_classes(ext, root_node, query_str)

    final_traits: list[dict[str, Any]] = []

    for item in all_classes:
        item_type = item.get('type', 'class')
        if item_type == 'trait':
            final_traits.append(item)

    return final_traits


def _scala_parse_all_classes(
    ext: GenericExtractor, root_node: Any, query_str: str
) -> list[dict[str, Any]]:
    """Parse all Scala class/object/trait definitions from a single query."""
    classes: list[dict[str, Any]] = []
    seen_nodes: set[tuple[int, int, str]] = set()

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "class":
            node_id = (node.start_byte, node.end_byte, node.type)
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)

            try:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                name_node = node.child_by_field_name("name")
                if name_node:
                    class_name = ext.get_node_text(name_node)
                    source_text = ext.get_node_text(node)

                    bases: list[str] = []
                    # Look for extends clause (extends_clause)
                    # class_definition -> extends_clause -> template_body
                    extends_clause = None
                    for child in node.children:
                        if child.type == "extends_clause":  # Might vary by grammar version: 'extends' keyword + types
                            extends_clause = child
                            break

                    if extends_clause:
                        for child in extends_clause.children:
                            if child.type == "type_identifier" or child.type == "user_type":  # specific to scala grammar
                                bases.append(ext.get_node_text(child))
                            elif child.type == "template_invocation":
                                # template_invocation -> user_type
                                pass

                    # Note: parsing bases in Scala can be complex (mixins with 'with' keyword).
                    # Using text based regex backup might be safer for now if tree query is hard.

                    class_data: dict[str, Any] = {
                        "name": class_name,
                        "line_number": start_line,
                        "end_line": end_line,
                        "bases": bases,
                        "path": ext.current_path,
                        "lang": ext.language_name,
                        "type": node.type.replace("_definition", ""),  # class, object, trait
                    }

                    if ext.index_source:
                        class_data["source"] = source_text

                    classes.append(class_data)

            except Exception as e:
                error_logger(f"Error parsing class in {ext.current_path}: {e}")
                continue

    return classes


def find_scala_imports(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Scala import declarations."""
    query_str = ext.config.queries.get("imports")
    if not query_str:
        return []

    imports: list[dict[str, Any]] = []

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "import":
            try:
                # Scala imports can be complex: import java.util.{Date, List} or import java.util._
                # We will try to extract the base path.
                import_text = ext.get_node_text(node)
                # Simple heuristic: remove 'import ' and handle one level
                clean_text = import_text.replace("import ", "").strip()

                # Split logic for multiple imports in one line not handled perfectly here yet
                # Just storing the whole text as name for now is better than crashing

                path = clean_text

                imports.append({
                    "name": path,
                    "full_import_name": path,
                    "line_number": node.start_point[0] + 1,
                    "alias": None,
                    "context": (None, None),
                    "lang": ext.language_name,
                    "is_dependency": False,
                })
            except Exception as e:
                error_logger(f"Error parsing import: {e}")
                continue

    return imports


def find_scala_calls(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Scala call expressions with field_expression handling and type inference."""
    query_str = ext.config.queries.get("calls")
    if not query_str:
        return []

    calls: list[dict[str, Any]] = []
    seen_calls: set[str] = set()

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "call_node":
            try:
                start_line = node.start_point[0] + 1

                # Heuristic to find name
                call_name = "unknown"
                full_name = "unknown"

                if node.type == "call_expression":
                    # function (child 0) arguments (child 1)
                    func_node = node.child_by_field_name("function")
                    if func_node:
                        if func_node.type == "field_expression":  # obj.method
                            call_name = ext.get_node_text(func_node.child_by_field_name("field"))  # or name?
                            full_name = ext.get_node_text(func_node)
                        elif func_node.type == "identifier":
                            call_name = ext.get_node_text(func_node)
                            full_name = call_name
                        elif func_node.type == "generic_function":
                            # generic_function -> function
                            inner = func_node.child_by_field_name("function")
                            if inner:
                                full_name = ext.get_node_text(inner)
                                call_name = full_name  # simplified

                if call_name == "unknown":
                    # Falback to text if simple
                    # call_name = ext.get_node_text(node).split('(')[0]
                    continue

                # Avoid duplicates
                call_key = f"{call_name}_{start_line}"
                if call_key in seen_calls:
                    continue
                seen_calls.add(call_key)

                ctx_name, ctx_type, ctx_line = _scala_get_parent_context(ext, node)

                # Inference from variables
                inferred_type = None
                if "." in full_name:
                    base_obj = full_name.split(".")[0]
                    # search for base_obj in variables
                    # Prefer variables in local context (ctx_name)

                    # Simple search: exact name match in same file
                    # We could improve by checking scope/context, but for now filtering by name is a good start
                    candidate = None
                    for v in ext._parsed_variables:
                        if v["name"] == base_obj:
                            # Check if context matches or is strictly enclosing?
                            # For now, just take the first match or last match?
                            # Usually last match (closest definition)
                            candidate = v
                            if v["context"] == ctx_name:
                                break

                    if candidate:
                        inferred_type = candidate["type"]
                elif call_name in ext._parsed_variables:  # Usually not happening as variables is list of dicts
                    pass

                calls.append({
                    "name": call_name,
                    "full_name": full_name,
                    "line_number": start_line,
                    "args": [],
                    "inferred_obj_type": inferred_type,
                    "context": (ctx_name, ctx_type, ctx_line),
                    "class_context": (ctx_name, ctx_line) if ctx_type and ("class" in str(ctx_type) or "object" in str(ctx_type)) else (None, None),
                    "lang": ext.language_name,
                    "is_dependency": False,
                })
            except Exception as e:
                error_logger(f"Error parsing call: {e}")
                continue

    return calls


def find_scala_variables(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Scala val/var definitions with type inference."""
    query_str = ext.config.queries.get("variables")
    if not query_str:
        return []

    variables: list[dict[str, Any]] = []
    seen_vars: set[int] = set()

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "variable":
            # The capture is on the whole definition (val/var_definition)
            # But we have @name on the identifier inside pattern.
            pass
        if capture_name == "name":
            # Check parent context
            if node.parent.type in ("val_definition", "var_definition"):
                definition = node.parent
                var_name = ext.get_node_text(node)
                start_line = node.start_point[0] + 1

                start_byte = node.start_byte
                if start_byte in seen_vars:
                    continue
                seen_vars.add(start_byte)

                ctx_name, ctx_type, ctx_line = _scala_get_parent_context(ext, node)

                # Type extraction: look for type_identifier in definition
                var_type = "Unknown"
                type_node = definition.child_by_field_name("type")
                if type_node:
                    var_type = ext.get_node_text(type_node)
                else:
                    # Attempt inference from value
                    val_node = definition.child_by_field_name("value")
                    if val_node:
                        if val_node.type == "instance_expression" or val_node.type == "new_expression":
                            # new Calculator()
                            # instance_expression -> new, type_identifier, arguments
                            for child in val_node.children:
                                if child.type in ("type_identifier", "simple_type", "user_type", "generic_type"):
                                    var_type = ext.get_node_text(child)
                                    break
                                elif child.type == "template_call":  # sometimes nested
                                    for sub in child.children:
                                        if sub.type in ("type_identifier", "simple_type", "user_type"):
                                            var_type = ext.get_node_text(sub)
                                            break
                        elif val_node.type == "call_expression":
                            # Circle(5.0)
                            # wrapper -> function(identifier)
                            func = val_node.child_by_field_name("function")
                            if func:
                                var_type = ext.get_node_text(func)

                variables.append({
                    "name": var_name,
                    "type": var_type,
                    "line_number": start_line,
                    "path": ext.current_path,
                    "lang": ext.language_name,
                    "context": ctx_name,
                    "class_context": ctx_name if ctx_type and ("class" in str(ctx_type) or "object" in str(ctx_type)) else None,
                })

    return variables


# ---------------------------------------------------------------------------
# Scala pre-scan
# ---------------------------------------------------------------------------

def pre_scan_scala(files: list[Path], parser_wrapper: Any) -> dict[str, list[str]]:
    """Pre-scan Scala files to build a map of class/object/trait names to file paths with FQN mapping."""
    name_to_files: dict[str, list[str]] = {}

    for path in files:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # package matches
            package_name = ""
            pkg_match = re.search(r'^\s*package\s+([\w\.]+)', content, re.MULTILINE)
            if pkg_match:
                package_name = pkg_match.group(1)

            # class/object/trait matches
            class_matches = re.finditer(r'\b(class|object|trait)\s+(\w+)', content)
            for match in class_matches:
                name = match.group(2)
                type_ = match.group(1)

                # Simple mapping
                if name not in name_to_files:
                    name_to_files[name] = []
                name_to_files[name].append(str(path))

                # FQN mapping
                if package_name:
                    fqn = f"{package_name}.{name}"
                    if fqn not in name_to_files:
                        name_to_files[fqn] = []
                    name_to_files[fqn].append(str(path))

        except Exception as e:
            error_logger(f"Error pre-scanning Scala file {path}: {e}")

    return name_to_files


# ============================================================================
# HASKELL
# ============================================================================
#
# NOTE: The Haskell parser is a broken copy-paste of the Kotlin parser.
# It uses Swift/Kotlin node types (function_declaration, class_declaration,
# struct_declaration, simple_identifier, navigation_expression, etc.) that
# DO NOT exist in Haskell's tree-sitter grammar. The pre_scan uses wrong
# keywords (class, object, interface, typealias). There is a bug where
# _parse_classes has `return classes` INSIDE a for loop (early return after
# first node). ALL bugs are preserved exactly as they appeared in the
# original source.
# ============================================================================

# ---------------------------------------------------------------------------
# Haskell helpers (module-level private)
# ---------------------------------------------------------------------------

def _haskell_get_parent_context(
    ext: GenericExtractor, node: Any
) -> tuple[str | None, str | None, int | None]:
    """Walk up AST using Haskell-specific context rules (copy-pasted from Kotlin, buggy)."""
    curr = node.parent
    while curr:
        if curr.type in ('class_declaration',):
            name_node = None
            for child in curr.children:
                if child.type == "simple_identifier":
                    name_node = child
                    break
                # BUG: return is INSIDE the for loop, returns on first child regardless
                return (
                    ext.get_node_text(name_node) if name_node else None,
                    curr.type,
                    curr.start_point[0] + 1,
                )
        if curr.type in ("class_declaration", "object_declaration"):
            for child in curr.children:
                if child.type in ("simple_identifier", "type_identifier"):
                    return (
                        ext.get_node_text(child),
                        curr.type,
                        curr.start_point[0] + 1,
                    )
            # check for secondary constructors
            if curr.type == "secondary_constructor":
                return (
                    "constructor",
                    curr.type,
                    curr.start_point[0] + 1,
                )
        if curr.type == "companion_object":
            name = "Companion"
            for child in curr.children:
                if child.type in ("simple_identifier", "type_identifier"):
                    name = ext.get_node_text(child)
                    break
            return (
                name,
                curr.type,
                curr.start_point[0] + 1,
            )
        # Handle anonymous objects (object_literal)
        if curr.type == "object_literal":
            # checking if it is assigned to a variable to get a name?
            # or simply "AnonymousObject"
            # it is usually hard to name them without variable context.

            name = "AnonymousObject"
            return (
                name,
                curr.type,
                curr.start_point[0] + 1,
            )
        curr = curr.parent
    return None, None, None


def _haskell_extract_parameter_names(params_text: str) -> list[str]:
    """Extract parameter names from a Kotlin-style parameter list (copy-pasted helper).

    This is the Kotlin parameter extraction logic that the Haskell parser
    was supposed to copy but didn't define. The original code calls
    self._extract_parameter_names which doesn't exist on the class -
    it would raise AttributeError at runtime if the code path were ever
    reached (it isn't, because the queries use wrong node types).
    """
    params: list[str] = []
    if not params_text:
        return params

    # Remove outer parentheses
    clean = params_text.strip()
    if clean.startswith('(') and clean.endswith(')'):
        clean = clean[1:-1]

    if not clean.strip():
        return params

    # Robust splitting by comma, respecting brackets <>, (), [], {}
    current_param: list[str] = []
    depth_angle = 0
    depth_round = 0
    depth_square = 0
    depth_curly = 0

    raw_params: list[str] = []

    for char in clean:
        if char == '<':
            depth_angle += 1
        elif char == '>':
            depth_angle -= 1
        elif char == '(':
            depth_round += 1
        elif char == ')':
            depth_round -= 1
        elif char == '[':
            depth_square += 1
        elif char == ']':
            depth_square -= 1
        elif char == '{':
            depth_curly += 1
        elif char == '}':
            depth_curly -= 1

        if char == ',' and depth_angle == 0 and depth_round == 0 and depth_square == 0 and depth_curly == 0:
            raw_params.append("".join(current_param).strip())
            current_param = []
        else:
            current_param.append(char)

    if current_param:
        raw_params.append("".join(current_param).strip())

    for p in raw_params:
        if not p:
            continue
        colon_index = p.find(':')
        if colon_index != -1:
            lhs = p[:colon_index].strip()
        else:
            lhs = p.strip()
        if not lhs:
            continue
        tokens = lhs.split()
        if tokens:
            params.append(tokens[-1])

    return params


# ---------------------------------------------------------------------------
# Haskell finder hooks
# ---------------------------------------------------------------------------

def find_haskell_functions(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Haskell functions (uses wrong Swift/Kotlin node types - bug preserved)."""
    query_str = ext.config.queries.get("functions")
    if not query_str:
        return []

    functions: list[dict[str, Any]] = []
    seen_nodes: set[tuple[int, int, str]] = set()

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "function_node":
            node_id = (node.start_byte, node.end_byte, node.type)
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)

            try:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                # Manual child lookup
                name_node = None
                for child in node.children:
                    if child.type == "simple_identifier":
                        name_node = child
                        break
                if name_node:
                    func_name = ext.get_node_text(name_node)

                    params_node = None
                    for child in node.children:
                        if child.type == "function_value_parameters":
                            params_node = child
                            break
                    parameters: list[str] = []
                    if params_node:
                        params_text = ext.get_node_text(params_node)
                        parameters = _haskell_extract_parameter_names(params_text)
                    source_text = ext.get_node_text(node)
                    context_name, context_type, context_line = _haskell_get_parent_context(ext, node)

                    func_data: dict[str, Any] = {
                        "name": func_name,
                        "args": parameters,
                        "line_number": start_line,
                        "end_line": end_line,
                        "path": ext.current_path,
                        "lang": ext.language_name,
                        "context": context_name,
                        "class_context": context_name if context_type and ("class" in context_type or "object" in context_type) else None,
                    }

                    if ext.index_source:
                        func_data["source"] = source_text

                    functions.append(func_data)
            except Exception as e:
                error_logger(f"Error parsing function in {ext.current_path}: {e}")
                continue

    return functions


def find_haskell_classes(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Haskell classes (uses wrong Swift node types, has early-return bug - all preserved)."""
    query_str = ext.config.queries.get("classes")
    if not query_str:
        return []

    classes: list[dict[str, Any]] = []
    seen_nodes: set[tuple[int, int, str]] = set()

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "class":
            node_id = (node.start_byte, node.end_byte, node.type)
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)

            try:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                # Find name child (type_identifier or simple_identifier)

                class_name = "Anonymous"
                if node.type == "companion_object":
                    class_name = "Companion"  # default name

                for child in node.children:
                    if child.type in ("type_identifier", "simple_identifier"):
                        class_name = ext.get_node_text(child)
                        break
                source_text = ext.get_node_text(node)

                bases: list[str] = []
                # Check for delegation specifiers
                # class_declaration -> delegation_specifier

                for child in node.children:
                    if child.type == "delegation_specifier":
                        # children: constructor_invocation or user_type -> type_identifier
                        # user_type -> type_identifier
                        for specifier in child.children:
                            # constructor_invocation -> user_type -> type_identifier
                            # user_type -> type_identifier
                            # We want text of the type
                            if specifier.type == "constructor_invocation":
                                # child 0 is typically user_type
                                for sub in specifier.children:
                                    if sub.type == "user_type":
                                        bases.append(ext.get_node_text(sub))
                            elif specifier.type == "user_type":
                                bases.append(ext.get_node_text(specifier))
                            elif specifier.type == "explicit_delegation":
                                # Not handling simple yet, uses 'by'
                                pass
                class_data: dict[str, Any] = {
                    "name": class_name,
                    "line_number": start_line,
                    "end_line": end_line,
                    "bases": bases,
                    "path": ext.current_path,
                    "lang": ext.language_name,
                }

                if ext.index_source:
                    class_data["source"] = source_text

                classes.append(class_data)
            except Exception as e:
                error_logger(f"Error parsing class in {ext.current_path}: {e}")
                continue
        # BUG PRESERVED: return inside the for loop - early return after first "class" capture
        return classes

    return classes


def find_haskell_imports(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Haskell imports (uses wrong node types - bug preserved)."""
    query_str = ext.config.queries.get("imports")
    if not query_str:
        return []

    imports: list[dict[str, Any]] = []

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "import":
            try:
                text = ext.get_node_text(node)
                # remove 'import'
                path = text.replace('import', '').strip().split(' as ')[0].strip()
                alias = None
                if ' as ' in text:
                    alias = text.split(' as ')[1].strip()
                imports.append({
                    "name": path,
                    "full_import_name": path,
                    "line_number": node.start_point[0] + 1,
                    "alias": alias,
                    "context": (None, None),
                    "lang": ext.language_name,
                    "is_dependency": False,
                })
            except Exception:
                continue

    return imports


def find_haskell_calls(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Haskell calls (uses Kotlin navigation_expression pattern - bug preserved)."""
    query_str = ext.config.queries.get("calls")
    if not query_str:
        return []

    calls: list[dict[str, Any]] = []
    seen_calls: set[str] = set()

    var_map: dict[tuple[str, str | None], str] = {}
    for v in ext._parsed_variables:
        key = (v['name'], v['context'])
        var_map[key] = v['type']

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "call_node":
            try:
                start_line = node.start_point[0] + 1
                func_name = "unknown"
                call_name = "unknown"
                base_obj = None

                children = node.children
                first_child = children[0]

                if first_child.type == "simple_identifier":
                    call_name = ext.get_node_text(first_child)
                elif first_child.type == "navigation_expression":
                    nav_children = first_child.children
                    if len(nav_children) >= 2:
                        operand = nav_children[0]
                        suffix = nav_children[-1]

                        if suffix.type == "navigation_suffix":
                            for c in suffix.children:
                                if c.type == "simple_identifier":
                                    call_name = ext.get_node_text(c)
                                    break
                        elif suffix.type == "simple_identifier":
                            call_name = ext.get_node_text(suffix)
                        base_obj = ext.get_node_text(operand)

                if call_name == "unknown":
                    continue
                full_name = f"{base_obj}.{call_name}" if base_obj else call_name

                ctx_name, ctx_type, ctx_line = _haskell_get_parent_context(ext, node)

                inferred_type = None
                if base_obj:
                    inferred_type = var_map.get((base_obj, ctx_name))
                    if not inferred_type:
                        inferred_type = var_map.get((base_obj, None))
                    if not inferred_type:
                        for (vname, vctx), vtype in var_map.items():
                            if vname == base_obj:
                                inferred_type = vtype
                                break

                calls.append({
                    "name": call_name,
                    "full_name": full_name,
                    "line_number": start_line,
                    "args": [],
                    "inferred_obj_type": inferred_type,
                    "context": [None, ctx_type, ctx_line],
                    "class_context": [None, None],
                    "lang": ext.language_name,
                    "is_dependency": False,
                })
            except Exception:
                continue

    return calls


def find_haskell_variables(ext: GenericExtractor, root_node: Any) -> list[dict[str, Any]]:
    """Extract Haskell variables (uses Kotlin property_declaration pattern - bug preserved)."""
    query_str = ext.config.queries.get("variables")
    if not query_str:
        return []

    variables: list[dict[str, Any]] = []
    seen_nodes: set[tuple[int, int, str]] = set()

    for node, capture_name in ext.run_query(query_str, root_node):
        if capture_name == "variable":
            try:
                start_line = node.start_point[0] + 1
                ctx_name, ctx_type, ctx_line = _haskell_get_parent_context(ext, node)

                # Destructuring declaration
                if "destructuring" in node.type:
                    pass
                # Regular property/variable
                var_name = "unknown"
                var_type = "Unknown"

                var_decl = None

                for child in node.children:
                    if child.type == "variable_declaration":
                        var_decl = child
                        break
                if var_decl:
                    # Check for name and type in variable_declaration
                    for child in var_decl.children:
                        if child.type == "simple_identifier":
                            var_name = ext.get_node_text(child)

                        if child.type == "user_type":
                            var_type = ext.get_node_text(child)

                # Attempt inference from initializer if type is unknown
                if var_type == "Unknown":
                    # property_declaration -> expression (e.g. call_expression)
                    for child in node.children:
                        if child.type == "call_expression":
                            # call_expression -> simple_identifier (constructor)
                            for sub in child.children:
                                if sub.type == "simple_identifier":
                                    var_type = ext.get_node_text(sub)
                                    break
                            if var_type != "Unknown":
                                break

                # BUG PRESERVED: comparison is `!= " unknown"` (with leading space) not `!= "unknown"`
                if var_name != " unknown":
                    variables.append({
                        "name": var_name,
                        "type": var_type,
                        "line_number": start_line,
                        "path": ext.current_path,
                        "lang": ext.language_name,
                        "context": ctx_name,
                        "class_context": ctx_name if ctx_type and ("class" in ctx_type or "object" in ctx_type) else None,
                    })
            except Exception:
                continue

    return variables


# ---------------------------------------------------------------------------
# Haskell pre-scan (uses wrong keywords: class, object, interface, typealias)
# ---------------------------------------------------------------------------

def pre_scan_haskell(files: list[Path], parser_wrapper: Any) -> dict[str, list[str]]:
    """Pre-scan Haskell files (uses Kotlin keywords - bug preserved)."""
    name_to_files: dict[str, list[str]] = {}
    for path in files:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            # 1. Extract package
            # package com.example.project
            package_name = ""
            pkg_match = re.search(r'^\s*package\s+([\w\.]+)', content, re.MULTILINE)
            if pkg_match:
                package_name = pkg_match.group(1)
            # 2. Extract classes/objects/interfaces/typealiases
            matches = re.finditer(
                r'^\s*(class|object|interface|typealias)\s+(\w+)',
                content,
            )
            for match in matches:
                name = match.group(2)
                # Map simple name
                if name not in name_to_files:
                    name_to_files[name] = []
                name_to_files[name].append(str(path))

                # If package exists, map FQN
                if package_name:
                    fqn = f"{package_name}.{name}"
                    if fqn not in name_to_files:
                        name_to_files[fqn] = []
                    name_to_files[fqn].append(str(path))
        except Exception:
            pass
    return name_to_files
