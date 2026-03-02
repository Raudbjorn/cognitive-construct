"""Graph builder — indexes code into SurrealDB embedded graph.

Parses source files via tree-sitter, creates nodes and graph edges
(contains, calls, inherits, imports, etc.) in SurrealQL.
"""

import asyncio
import os
import pathspec
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime

from ..database import DatabaseManager
from .jobs import JobManager, JobStatus
from .debug_log import debug_log, info_logger, error_logger, warning_logger

from tree_sitter import Language, Parser
from .tree_sitter_manager import get_tree_sitter_manager
from .config import get_config_value

# Node types that get is_searchable=true for vector index support
_SEARCHABLE_TYPES = {"Function", "Class", "Variable"}


class TreeSitterParser:
    """A generic parser wrapper for a specific language using tree-sitter."""

    def __init__(self, language_name: str):
        from .languages import GenericExtractor, get_config

        self.language_name = language_name
        self.ts_manager = get_tree_sitter_manager()

        self.language: Language = self.ts_manager.get_language_safe(language_name)
        self.parser = Parser(self.language)

        config = get_config(language_name)
        self.extractor = GenericExtractor(config, self)

    def parse(self, path: Path, is_dependency: bool = False, **kwargs) -> Dict:
        """Dispatches parsing to the config-driven GenericExtractor."""
        return self.extractor.parse(path, is_dependency, **kwargs)


class GraphBuilder:
    """Module for building and managing the SurrealDB code graph."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        job_manager: JobManager,
        loop: asyncio.AbstractEventLoop,
    ):
        self.db_manager = db_manager
        self.job_manager = job_manager
        self.loop = loop
        self.parsers = {
            '.py': TreeSitterParser('python'),
            '.ipynb': TreeSitterParser('python'),
            '.js': TreeSitterParser('javascript'),
            '.jsx': TreeSitterParser('javascript'),
            '.mjs': TreeSitterParser('javascript'),
            '.cjs': TreeSitterParser('javascript'),
            '.go': TreeSitterParser('go'),
            '.ts': TreeSitterParser('typescript'),
            '.tsx': TreeSitterParser('typescript'),
            '.cpp': TreeSitterParser('cpp'),
            '.h': TreeSitterParser('cpp'),
            '.hpp': TreeSitterParser('cpp'),
            '.hh': TreeSitterParser('cpp'),
            '.rs': TreeSitterParser('rust'),
            '.c': TreeSitterParser('c'),
            '.java': TreeSitterParser('java'),
            '.rb': TreeSitterParser('ruby'),
            '.cs': TreeSitterParser('c_sharp'),
            '.php': TreeSitterParser('php'),
            '.kt': TreeSitterParser('kotlin'),
            '.scala': TreeSitterParser('scala'),
            '.sc': TreeSitterParser('scala'),
            '.swift': TreeSitterParser('swift'),
            '.hs': TreeSitterParser('haskell'),
        }

    # ------------------------------------------------------------------
    # Pre-scan for imports
    # ------------------------------------------------------------------

    def _pre_scan_for_imports(self, files: list[Path]) -> dict:
        """Dispatches pre-scan to the correct language-specific implementation."""
        imports_map: dict[str, list[str]] = {}

        files_by_lang: dict[str, list[Path]] = {}
        for file in files:
            if file.suffix in self.parsers:
                files_by_lang.setdefault(file.suffix, []).append(file)

        for ext, lang_files in files_by_lang.items():
            parser = self.parsers[ext]
            scan_fn = parser.extractor.config.pre_scan_fn
            if not scan_fn:
                continue
            try:
                imports_map.update(scan_fn(lang_files, parser))
            except Exception:
                debug_log(f"Pre-scan failed for {ext}")

        return imports_map

    # ------------------------------------------------------------------
    # Repository
    # ------------------------------------------------------------------

    async def add_repository_to_graph(
        self, repo_path: Path, is_dependency: bool = False
    ) -> str | None:
        """Create or update a repository node. Returns the record id."""
        repo_name = repo_path.name
        repo_path_str = str(repo_path.resolve())
        result = await self.db_manager.query(
            """
            UPSERT node SET
                node_type = 'Repository',
                name = $name,
                path = $path,
                is_dependency = $is_dep
            WHERE node_type = 'Repository' AND path = $path;
            """,
            {"name": repo_name, "path": repo_path_str, "is_dep": is_dependency},
        )

        # Also create in the repository table for fast lookups
        await self.db_manager.query(
            """
            UPSERT repository SET
                name = $name,
                path = $path,
                is_dependency = $is_dep
            WHERE path = $path;
            """,
            {"name": repo_name, "path": repo_path_str, "is_dep": is_dependency},
        )

        # Return the node id for CONTAINS edges
        res = await self.db_manager.query(
            "SELECT id FROM node WHERE node_type = 'Repository' AND path = $path LIMIT 1;",
            {"path": repo_path_str},
        )
        rows = [r for r in (res or []) if isinstance(r, dict)]
        if rows:
            return rows[0].get("id")
        return None

    # ------------------------------------------------------------------
    # File + contents (first pass)
    # ------------------------------------------------------------------

    async def add_file_to_graph(
        self, file_data: Dict, repo_name: str, imports_map: dict
    ) -> None:
        """Add a file and its contents (functions, classes, variables, etc.)."""
        db = self.db_manager
        file_path_str = str(Path(file_data['path']).resolve())
        file_name = Path(file_path_str).name
        is_dependency = file_data.get('is_dependency', False)

        # Resolve relative path from repo
        repo_path_str = str(Path(file_data['repo_path']).resolve())
        try:
            relative_path = str(Path(file_path_str).relative_to(Path(repo_path_str)))
        except ValueError:
            relative_path = file_name

        # Create File node
        await db.query(
            """
            UPSERT node SET
                node_type = 'File',
                name = $name,
                path = $path,
                relative_path = $rel_path,
                is_dependency = $is_dep,
                language = $lang
            WHERE node_type = 'File' AND path = $path;
            """,
            {
                "name": file_name,
                "path": file_path_str,
                "rel_path": relative_path,
                "is_dep": is_dependency,
                "lang": file_data.get('lang'),
            },
        )

        # Build directory chain: Repository → Directory → ... → File
        file_path_obj = Path(file_path_str)
        repo_path_obj = Path(repo_path_str)

        try:
            relative_path_to_file = file_path_obj.relative_to(repo_path_obj)
        except ValueError:
            relative_path_to_file = Path(file_name)

        parent_type = 'Repository'
        parent_path = str(repo_path_obj)

        for part in relative_path_to_file.parts[:-1]:
            current_path = str(Path(parent_path) / part)

            # Create Directory node
            await db.query(
                """
                UPSERT node SET
                    node_type = 'Directory',
                    name = $name,
                    path = $dir_path
                WHERE node_type = 'Directory' AND path = $dir_path;
                """,
                {"name": part, "dir_path": current_path},
            )

            # CONTAINS: parent → directory
            await self._relate_by_path(
                parent_type, parent_path, 'Directory', current_path, 'contains'
            )

            parent_path = current_path
            parent_type = 'Directory'

        # CONTAINS: parent → file
        await self._relate_by_path(
            parent_type, parent_path, 'File', file_path_str, 'contains'
        )

        # ------------------------------------------------------------------
        # Node items: functions, classes, variables, etc.
        # ------------------------------------------------------------------
        item_mappings = [
            (file_data.get('functions', []), 'Function'),
            (file_data.get('classes', []), 'Class'),
            (file_data.get('traits', []), 'Trait'),
            (file_data.get('variables', []), 'Variable'),
            (file_data.get('interfaces', []), 'Interface'),
            (file_data.get('macros', []), 'Macro'),
            (file_data.get('structs', []), 'Struct'),
            (file_data.get('enums', []), 'Enum'),
            (file_data.get('unions', []), 'Union'),
            (file_data.get('records', []), 'Record'),
            (file_data.get('properties', []), 'Property'),
        ]

        for item_data, label in item_mappings:
            for item in item_data:
                if label == 'Function' and 'cyclomatic_complexity' not in item:
                    item['cyclomatic_complexity'] = 1

                is_searchable = label in _SEARCHABLE_TYPES
                line_number = item.get('line_number', 0)

                await db.query(
                    """
                    UPSERT node SET
                        node_type = $ntype,
                        name = $name,
                        path = $path,
                        line_number = $line,
                        end_line = $end_line,
                        source = $source,
                        docstring = $docstring,
                        language = $lang,
                        is_dependency = $is_dep,
                        is_searchable = $searchable,
                        args = $args,
                        decorators = $decorators,
                        complexity = $complexity,
                        bases = $bases,
                        value = $value,
                        context = $context
                    WHERE node_type = $ntype AND name = $name
                        AND path = $path AND line_number = $line;
                    """,
                    {
                        "ntype": label,
                        "name": item['name'],
                        "path": file_path_str,
                        "line": line_number,
                        "end_line": item.get('end_line_number'),
                        "source": item.get('source'),
                        "docstring": item.get('docstring'),
                        "lang": file_data.get('lang'),
                        "is_dep": is_dependency,
                        "searchable": is_searchable,
                        "args": item.get('args'),
                        "decorators": item.get('decorators'),
                        "complexity": item.get('cyclomatic_complexity'),
                        "bases": item.get('bases'),
                        "value": item.get('value'),
                        "context": item.get('context'),
                    },
                )

                # CONTAINS: File → node
                await self._relate_by_type_path_line(
                    'File', file_path_str, None,
                    label, file_path_str, line_number,
                    'contains',
                )

                # Parameters
                if label == 'Function':
                    for arg_name in item.get('args', []):
                        await db.query(
                            """
                            UPSERT node SET
                                node_type = 'Parameter',
                                name = $arg_name,
                                path = $path,
                                line_number = $func_line
                            WHERE node_type = 'Parameter' AND name = $arg_name
                                AND path = $path AND line_number = $func_line;
                            """,
                            {
                                "arg_name": arg_name,
                                "path": file_path_str,
                                "func_line": line_number,
                            },
                        )
                        await self._relate_by_type_path_line(
                            'Function', file_path_str, line_number,
                            'Parameter', file_path_str, line_number,
                            'has_parameter',
                        )

        # Ruby modules
        for m in file_data.get('modules', []):
            await db.query(
                """
                UPSERT node SET
                    node_type = 'Module',
                    name = $name,
                    language = $lang
                WHERE node_type = 'Module' AND name = $name;
                """,
                {"name": m["name"], "lang": file_data.get("lang")},
            )

        # Nested function CONTAINS
        for item in file_data.get('functions', []):
            if item.get("context_type") == "function_definition":
                await self._relate_nested_function(
                    file_path_str, item["context"], item["name"], item["line_number"]
                )

        # Imports
        for imp in file_data.get('imports', []):
            lang = file_data.get('lang')
            if lang == 'javascript':
                module_name = imp.get('source')
                if not module_name:
                    continue

                await db.query(
                    """
                    UPSERT node SET
                        node_type = 'Module',
                        name = $module_name
                    WHERE node_type = 'Module' AND name = $module_name;
                    """,
                    {"module_name": module_name},
                )

                rel_props = {"imported_name": imp.get('name', '*')}
                if imp.get('alias'):
                    rel_props['alias'] = imp['alias']
                if imp.get('line_number'):
                    rel_props['line_number'] = imp['line_number']

                await self._relate_file_imports_module(
                    file_path_str, module_name, rel_props
                )
            else:
                imp_name = imp.get('name', '')
                await db.query(
                    """
                    UPSERT node SET
                        node_type = 'Module',
                        name = $name,
                        alias = $alias,
                        full_import_name = $full_name
                    WHERE node_type = 'Module' AND name = $name;
                    """,
                    {
                        "name": imp_name,
                        "alias": imp.get('alias'),
                        "full_name": imp.get('full_import_name'),
                    },
                )

                rel_props: dict[str, Any] = {}
                if imp.get('line_number'):
                    rel_props['line_number'] = imp['line_number']
                if imp.get('alias'):
                    rel_props['alias'] = imp['alias']

                await self._relate_file_imports_module(
                    file_path_str, imp_name, rel_props
                )

        # Class → Function (CONTAINS for methods)
        for func in file_data.get('functions', []):
            if func.get('class_context'):
                await self._relate_class_contains_method(
                    file_path_str, func['class_context'], func['name'], func['line_number']
                )

        # Ruby module inclusions (Class INCLUDES Module)
        for inc in file_data.get('module_inclusions', []):
            await self._relate_class_includes_module(
                file_path_str, inc["class"], inc["module"]
            )

    # ------------------------------------------------------------------
    # Relationship helpers
    # ------------------------------------------------------------------

    async def _relate_by_path(
        self,
        from_type: str, from_path: str,
        to_type: str, to_path: str,
        relation: str,
    ) -> None:
        """Create a relation between two nodes matched by node_type + path."""
        await self.db_manager.query(
            f"""
            LET $from = (SELECT id FROM node
                WHERE node_type = $ftype AND path = $fpath LIMIT 1);
            LET $to = (SELECT id FROM node
                WHERE node_type = $ttype AND path = $tpath LIMIT 1);
            IF array::len($from) > 0 AND array::len($to) > 0 {{
                RELATE $from->{relation}->$to;
            }};
            """,
            {
                "ftype": from_type, "fpath": from_path,
                "ttype": to_type, "tpath": to_path,
            },
        )

    async def _relate_by_type_path_line(
        self,
        from_type: str, from_path: str, from_line: int | None,
        to_type: str, to_path: str, to_line: int | None,
        relation: str,
    ) -> None:
        """Create a relation between two nodes matched by type+path+line."""
        if from_line is not None:
            from_clause = (
                "SELECT id FROM node WHERE node_type = $ftype "
                "AND path = $fpath AND line_number = $fline LIMIT 1"
            )
        else:
            from_clause = (
                "SELECT id FROM node WHERE node_type = $ftype "
                "AND path = $fpath LIMIT 1"
            )

        if to_line is not None:
            to_clause = (
                "SELECT id FROM node WHERE node_type = $ttype "
                "AND path = $tpath AND line_number = $tline LIMIT 1"
            )
        else:
            to_clause = (
                "SELECT id FROM node WHERE node_type = $ttype "
                "AND path = $tpath LIMIT 1"
            )

        await self.db_manager.query(
            f"""
            LET $from = ({from_clause});
            LET $to = ({to_clause});
            IF array::len($from) > 0 AND array::len($to) > 0 {{
                RELATE $from->{relation}->$to;
            }};
            """,
            {
                "ftype": from_type, "fpath": from_path, "fline": from_line,
                "ttype": to_type, "tpath": to_path, "tline": to_line,
            },
        )

    async def _relate_nested_function(
        self, file_path: str, outer_name: str, inner_name: str, inner_line: int
    ) -> None:
        await self.db_manager.query(
            """
            LET $outer = (SELECT id FROM node
                WHERE node_type = 'Function' AND name = $outer AND path = $path LIMIT 1);
            LET $inner = (SELECT id FROM node
                WHERE node_type = 'Function' AND name = $inner
                AND path = $path AND line_number = $line LIMIT 1);
            IF array::len($outer) > 0 AND array::len($inner) > 0 {
                RELATE $outer->contains->$inner;
            };
            """,
            {"outer": outer_name, "inner": inner_name, "path": file_path, "line": inner_line},
        )

    async def _relate_file_imports_module(
        self, file_path: str, module_name: str, props: dict
    ) -> None:
        set_clause = ""
        if props:
            assignments = ", ".join(f"{k} = ${k}" for k in props)
            set_clause = f"SET {assignments}"

        params = {
            "fpath": file_path,
            "mod_name": module_name,
            **props,
        }
        await self.db_manager.query(
            f"""
            LET $file = (SELECT id FROM node
                WHERE node_type = 'File' AND path = $fpath LIMIT 1);
            LET $mod = (SELECT id FROM node
                WHERE node_type = 'Module' AND name = $mod_name LIMIT 1);
            IF array::len($file) > 0 AND array::len($mod) > 0 {{
                RELATE $file->imports->$mod {set_clause};
            }};
            """,
            params,
        )

    async def _relate_class_contains_method(
        self, file_path: str, class_name: str, func_name: str, func_line: int
    ) -> None:
        await self.db_manager.query(
            """
            LET $cls = (SELECT id FROM node
                WHERE node_type = 'Class' AND name = $cname AND path = $path LIMIT 1);
            LET $fn = (SELECT id FROM node
                WHERE node_type = 'Function' AND name = $fname
                AND path = $path AND line_number = $fline LIMIT 1);
            IF array::len($cls) > 0 AND array::len($fn) > 0 {
                RELATE $cls->contains->$fn;
            };
            """,
            {"cname": class_name, "fname": func_name, "path": file_path, "fline": func_line},
        )

    async def _relate_class_includes_module(
        self, file_path: str, class_name: str, module_name: str
    ) -> None:
        await self.db_manager.query(
            """
            LET $cls = (SELECT id FROM node
                WHERE node_type = 'Class' AND name = $cname AND path = $path LIMIT 1);
            LET $mod = (SELECT id FROM node
                WHERE node_type = 'Module' AND name = $mname LIMIT 1);
            IF array::len($cls) > 0 AND array::len($mod) > 0 {
                RELATE $cls->includes->$mod;
            };
            """,
            {"cname": class_name, "mname": module_name, "path": file_path},
        )

    # ------------------------------------------------------------------
    # Second pass: function calls
    # ------------------------------------------------------------------

    async def _create_function_calls(self, file_data: Dict, imports_map: dict) -> None:
        """Create CALLS relationships with prioritized resolution logic."""
        caller_file_path = str(Path(file_data['path']).resolve())
        local_names = (
            {f['name'] for f in file_data.get('functions', [])}
            | {c['name'] for c in file_data.get('classes', [])}
        )
        local_imports = {
            imp.get('alias') or imp['name'].split('.')[-1]: imp['name']
            for imp in file_data.get('imports', [])
        }

        for call in file_data.get('function_calls', []):
            called_name = call['name']
            if called_name in __builtins__:
                continue

            resolved_path = None
            full_call = call.get('full_name', called_name)
            base_obj = full_call.split('.')[0] if '.' in full_call else None
            is_chained_call = full_call.count('.') > 1 if '.' in full_call else False

            if is_chained_call and base_obj in ('self', 'this', 'super', 'super()', 'cls', '@'):
                lookup_name = called_name
            else:
                lookup_name = base_obj if base_obj else called_name

            # 1. Local context keywords / direct local names
            if base_obj in ('self', 'this', 'super', 'super()', 'cls', '@') and not is_chained_call:
                resolved_path = caller_file_path
            elif lookup_name in local_names:
                resolved_path = caller_file_path

            # 2. Inferred type
            elif call.get('inferred_obj_type'):
                obj_type = call['inferred_obj_type']
                possible_paths = imports_map.get(obj_type, [])
                if possible_paths:
                    resolved_path = possible_paths[0]

            # 3. Imports map
            if not resolved_path:
                possible_paths = imports_map.get(lookup_name, [])
                if len(possible_paths) == 1:
                    resolved_path = possible_paths[0]
                elif len(possible_paths) > 1:
                    if lookup_name in local_imports:
                        full_import_name = local_imports[lookup_name]
                        if full_import_name in imports_map:
                            direct_paths = imports_map[full_import_name]
                            if direct_paths and len(direct_paths) == 1:
                                resolved_path = direct_paths[0]
                        if not resolved_path:
                            for path in possible_paths:
                                if full_import_name.replace('.', '/') in path:
                                    resolved_path = path
                                    break

            # 4. Final fallback
            if not resolved_path:
                if called_name in local_names:
                    resolved_path = caller_file_path
                elif called_name in imports_map and imports_map[called_name]:
                    candidates = imports_map[called_name]
                    for path in candidates:
                        for imp_name in local_imports.values():
                            if imp_name.replace('.', '/') in path:
                                resolved_path = path
                                break
                        if resolved_path:
                            break
                    if not resolved_path:
                        resolved_path = candidates[0]
                else:
                    resolved_path = caller_file_path

            caller_context = call.get('context')
            if caller_context and len(caller_context) == 3 and caller_context[0] is not None:
                caller_name, _, caller_line_number = caller_context
                await self._create_call_edge(
                    caller_name=caller_name,
                    caller_file_path=caller_file_path,
                    caller_line_number=caller_line_number,
                    called_name=called_name,
                    called_file_path=resolved_path,
                    call_line_number=call['line_number'],
                    args=call.get('args', []),
                    full_call_name=call.get('full_name', called_name),
                    caller_is_file=False,
                )
            else:
                await self._create_call_edge(
                    caller_name=None,
                    caller_file_path=caller_file_path,
                    caller_line_number=None,
                    called_name=called_name,
                    called_file_path=resolved_path,
                    call_line_number=call['line_number'],
                    args=call.get('args', []),
                    full_call_name=call.get('full_name', called_name),
                    caller_is_file=True,
                )

    async def _create_call_edge(
        self,
        caller_name: str | None,
        caller_file_path: str,
        caller_line_number: int | None,
        called_name: str,
        called_file_path: str,
        call_line_number: int,
        args: list,
        full_call_name: str,
        caller_is_file: bool,
    ) -> None:
        """Create a CALLS edge with __init__/constructor redirect for classes."""
        if caller_is_file:
            caller_query = (
                "SELECT id FROM node WHERE node_type = 'File' "
                "AND path = $caller_path LIMIT 1"
            )
        else:
            caller_query = (
                "SELECT id FROM node WHERE (node_type = 'Function' OR node_type = 'Class') "
                "AND name = $caller_name AND path = $caller_path "
                "AND line_number = $caller_line LIMIT 1"
            )

        # Split into two queries to avoid $var[0].field indexing in SurrealDB.
        # First: find the called node.
        called_result = await self.db_manager.query(
            """
            SELECT id, node_type FROM node
            WHERE (node_type = 'Function' OR node_type = 'Class')
                AND name = $called_name AND path = $called_path LIMIT 1;
            """,
            {"called_name": called_name, "called_path": called_file_path},
        )

        called_rows = [r for r in (called_result or []) if isinstance(r, dict)]
        if not called_rows:
            return

        called_node = called_rows[0]
        target_id = called_node["id"]

        # If called node is a Class, redirect to __init__/constructor if it exists
        if called_node["node_type"] == "Class":
            init_result = await self.db_manager.query(
                """
                SELECT id FROM node
                WHERE (name = '__init__' OR name = 'constructor')
                    AND node_type = 'Function' AND path = $called_path
                    AND context = $called_name LIMIT 1;
                """,
                {"called_path": called_file_path, "called_name": called_name},
            )
            init_rows = [r for r in (init_result or []) if isinstance(r, dict)]
            if init_rows:
                target_id = init_rows[0]["id"]

        # Now create the CALLS edge
        await self.db_manager.query(
            f"""
            LET $caller = ({caller_query});
            IF array::len($caller) > 0 {{
                RELATE $caller->calls->$target
                    SET line_number = $call_line,
                        args = $args,
                        full_call_name = $fcn;
            }};
            """,
            {
                "caller_name": caller_name,
                "caller_path": caller_file_path,
                "caller_line": caller_line_number,
                "target": target_id,
                "call_line": call_line_number,
                "args": args,
                "fcn": full_call_name,
            },
        )

    async def _create_all_function_calls(
        self, all_file_data: list[Dict], imports_map: dict
    ) -> None:
        for file_data in all_file_data:
            await self._create_function_calls(file_data, imports_map)

    # ------------------------------------------------------------------
    # Second pass: inheritance
    # ------------------------------------------------------------------

    async def _create_inheritance_links(
        self, file_data: Dict, imports_map: dict
    ) -> None:
        caller_file_path = str(Path(file_data['path']).resolve())
        local_class_names = {c['name'] for c in file_data.get('classes', [])}
        local_imports = {
            imp.get('alias') or imp['name'].split('.')[-1]: imp['name']
            for imp in file_data.get('imports', [])
        }

        for class_item in file_data.get('classes', []):
            if not class_item.get('bases'):
                continue

            for base_class_str in class_item['bases']:
                if base_class_str == 'object':
                    continue

                resolved_path = None
                target_class_name = base_class_str.split('.')[-1]

                if '.' in base_class_str:
                    lookup_name = base_class_str.split('.')[0]
                    if lookup_name in local_imports:
                        full_import_name = local_imports[lookup_name]
                        possible_paths = imports_map.get(target_class_name, [])
                        for path in possible_paths:
                            if full_import_name.replace('.', '/') in path:
                                resolved_path = path
                                break
                else:
                    lookup_name = base_class_str
                    if lookup_name in local_class_names:
                        resolved_path = caller_file_path
                    elif lookup_name in local_imports:
                        full_import_name = local_imports[lookup_name]
                        possible_paths = imports_map.get(target_class_name, [])
                        for path in possible_paths:
                            if full_import_name.replace('.', '/') in path:
                                resolved_path = path
                                break
                    elif lookup_name in imports_map:
                        possible_paths = imports_map[lookup_name]
                        if len(possible_paths) == 1:
                            resolved_path = possible_paths[0]

                if resolved_path:
                    await self.db_manager.query(
                        """
                        LET $child = (SELECT id FROM node
                            WHERE node_type = 'Class' AND name = $child_name
                            AND path = $path LIMIT 1);
                        LET $parent = (SELECT id FROM node
                            WHERE node_type = 'Class' AND name = $parent_name
                            AND path = $parent_path LIMIT 1);
                        IF array::len($child) > 0 AND array::len($parent) > 0 {
                            RELATE $child->inherits->$parent;
                        };
                        """,
                        {
                            "child_name": class_item['name'],
                            "path": caller_file_path,
                            "parent_name": target_class_name,
                            "parent_path": resolved_path,
                        },
                    )

    async def _create_csharp_inheritance_and_interfaces(
        self, file_data: Dict, imports_map: dict
    ) -> None:
        if file_data.get('lang') != 'c_sharp':
            return

        caller_file_path = str(Path(file_data['path']).resolve())
        local_type_names: set[str] = set()
        for type_list in ['classes', 'interfaces', 'structs', 'records']:
            local_type_names.update(t['name'] for t in file_data.get(type_list, []))

        for type_list_name, type_label in [
            ('classes', 'Class'), ('structs', 'Struct'),
            ('records', 'Record'), ('interfaces', 'Interface'),
        ]:
            for type_item in file_data.get(type_list_name, []):
                if not type_item.get('bases'):
                    continue

                for base_str in type_item['bases']:
                    base_name = base_str.split('<')[0].strip()
                    is_interface = any(
                        iface['name'] == base_name
                        for iface in file_data.get('interfaces', [])
                    )
                    resolved_path = caller_file_path
                    if base_name in imports_map:
                        possible_paths = imports_map[base_name]
                        if possible_paths:
                            resolved_path = possible_paths[0]

                    base_index = type_item['bases'].index(base_str)
                    relation = (
                        'implements'
                        if is_interface or (base_index > 0 and type_label == 'Class')
                        else 'inherits'
                    )

                    await self.db_manager.query(
                        f"""
                        LET $child = (SELECT id FROM node
                            WHERE name = $child_name AND path = $path
                            AND node_type IN ['Class', 'Struct', 'Record', 'Interface'] LIMIT 1);
                        LET $parent = (SELECT id FROM node
                            WHERE name = $parent_name
                            AND node_type IN ['Class', 'Struct', 'Record', 'Interface'] LIMIT 1);
                        IF array::len($child) > 0 AND array::len($parent) > 0 {{
                            RELATE $child->{relation}->$parent;
                        }};
                        """,
                        {
                            "child_name": type_item['name'],
                            "path": caller_file_path,
                            "parent_name": base_name,
                        },
                    )

    async def _create_all_inheritance_links(
        self, all_file_data: list[Dict], imports_map: dict
    ) -> None:
        for file_data in all_file_data:
            if file_data.get('lang') == 'c_sharp':
                await self._create_csharp_inheritance_and_interfaces(file_data, imports_map)
            else:
                await self._create_inheritance_links(file_data, imports_map)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def _embed_nodes(self, repo_name: str, job_id: str | None = None) -> None:
        """Compute and store embeddings for searchable nodes missing them."""
        if os.environ.get("CODEGRAPH_SKIP_EMBEDDINGS", "").strip() == "1":
            info_logger("Embedding pass skipped (CODEGRAPH_SKIP_EMBEDDINGS=1)")
            return

        try:
            from ..embeddings import compose_embedding_text, encode_texts, is_available

            if not is_available():
                info_logger("sentence-transformers not installed; skipping embeddings")
                return

            result = await self.db_manager.query(
                """
                SELECT id, node_type, name, docstring, source, path
                FROM node
                WHERE is_searchable = true AND embedding = NONE;
                """
            )

            records = [r for r in (result or []) if isinstance(r, dict)]
            if not records:
                info_logger("No nodes need embedding")
                return

            info_logger(f"Embedding {len(records)} nodes...")
            if job_id:
                self.job_manager.update_job(
                    job_id, current_file=f"Embedding {len(records)} nodes"
                )

            texts: list[str] = []
            ids: list[str] = []
            for rec in records:
                text = compose_embedding_text(
                    name=rec.get("name", ""),
                    docstring=rec.get("docstring"),
                    source=rec.get("source"),
                    context=rec.get("path"),
                    node_type=rec.get("node_type", "Code"),
                )
                texts.append(text)
                ids.append(rec["id"])

            vectors = encode_texts(texts)
            if vectors is None:
                warning_logger("encode_texts returned None; skipping embedding write")
                return

            batch_size = 100
            for i in range(0, len(ids), batch_size):
                for j in range(i, min(i + batch_size, len(ids))):
                    await self.db_manager.query(
                        "UPDATE $id SET embedding = $vec;",
                        {"id": ids[j], "vec": vectors[j].tolist()},
                    )

            info_logger(f"Embedded {len(ids)} nodes for repo '{repo_name}'")

        except Exception:
            warning_logger("Embedding pass failed (non-blocking)")
            import traceback
            warning_logger(traceback.format_exc())

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    async def delete_file_from_graph(self, path: str) -> None:
        file_path_str = str(Path(path).resolve())

        # Get parent directories before deleting
        parent_result = await self.db_manager.query(
            """
            SELECT path FROM node
            WHERE node_type = 'Directory'
            AND path IN (
                SELECT VALUE <-contains<-node.path FROM node
                WHERE node_type = 'File' AND path = $path
            )
            ORDER BY path DESC;
            """,
            {"path": file_path_str},
        )
        parent_paths = [r["path"] for r in (parent_result or []) if isinstance(r, dict) and "path" in r]

        # Delete file's contained elements
        await self.db_manager.query(
            """
            DELETE node WHERE path = $path AND node_type != 'File' AND node_type != 'Directory';
            DELETE node WHERE node_type = 'File' AND path = $path;
            """,
            {"path": file_path_str},
        )
        info_logger(f"Deleted file and its elements from graph: {file_path_str}")

        # Clean up empty directories
        for dir_path in parent_paths:
            await self.db_manager.query(
                """
                LET $children = (SELECT id FROM node
                    WHERE node_type = 'Directory' AND path = $path
                    AND count(->contains->node) > 0);
                IF array::len($children) = 0 {
                    DELETE node WHERE node_type = 'Directory' AND path = $path;
                };
                """,
                {"path": dir_path},
            )

    async def delete_repository_from_graph(self, repo_path: str) -> bool:
        repo_path_str = str(Path(repo_path).resolve())

        check = await self.db_manager.query(
            "SELECT count() AS cnt FROM node WHERE node_type = 'Repository' AND path = $path GROUP ALL;",
            {"path": repo_path_str},
        )
        rows = check if isinstance(check, list) else []
        rows = [r for r in rows if isinstance(r, dict)]
        cnt = rows[0]["cnt"] if rows else 0
        if cnt == 0:
            warning_logger(f"Attempted to delete non-existent repository: {repo_path_str}")
            return False

        # Delete all nodes belonging to this repo path
        # (nodes whose path starts with the repo path, or who are the repo itself)
        await self.db_manager.query(
            """
            DELETE node WHERE path != NONE AND string::starts_with(path, $path);
            DELETE node WHERE node_type = 'Repository' AND path = $path;
            DELETE repository WHERE path = $path;
            """,
            {"path": repo_path_str},
        )
        info_logger(f"Deleted repository and its contents from graph: {repo_path_str}")
        return True

    # ------------------------------------------------------------------
    # File parsing
    # ------------------------------------------------------------------

    def parse_file(
        self, repo_path: Path, path: Path, is_dependency: bool = False
    ) -> Dict:
        parser = self.parsers.get(path.suffix)
        if not parser:
            warning_logger(f"No parser found for file extension {path.suffix}. Skipping {path}")
            return {"path": str(path), "error": f"No parser for {path.suffix}"}

        debug_log(f"[parse_file] Starting parsing for: {path} with {parser.language_name} parser")
        try:
            index_source = (get_config_value("INDEX_SOURCE") or "false").lower() == "true"
            if parser.language_name == 'python':
                is_notebook = path.suffix == '.ipynb'
                file_data = parser.parse(
                    path, is_dependency,
                    is_notebook=is_notebook,
                    index_source=index_source,
                )
            else:
                file_data = parser.parse(
                    path, is_dependency,
                    index_source=index_source,
                )
            file_data['repo_path'] = str(repo_path)
            return file_data
        except Exception as e:
            error_logger(f"Error parsing {path} with {parser.language_name} parser: {e}")
            return {"path": str(path), "error": str(e)}

    def estimate_processing_time(self, path: Path) -> Optional[Tuple[int, float]]:
        try:
            supported_extensions = self.parsers.keys()
            if path.is_file():
                if path.suffix in supported_extensions:
                    files = [path]
                else:
                    return 0, 0.0
            else:
                all_files = path.rglob("*")
                files = [f for f in all_files if f.is_file() and f.suffix in supported_extensions]

                ignore_dirs_str = get_config_value("IGNORE_DIRS") or ""
                if ignore_dirs_str:
                    ignore_dirs = {d.strip().lower() for d in ignore_dirs_str.split(',') if d.strip()}
                    if ignore_dirs:
                        kept_files = []
                        for f in files:
                            try:
                                parts = set(p.lower() for p in f.relative_to(path).parent.parts)
                                if not parts.intersection(ignore_dirs):
                                    kept_files.append(f)
                            except ValueError:
                                kept_files.append(f)
                        files = kept_files

            total_files = len(files)
            estimated_time = total_files * 0.05
            return total_files, estimated_time
        except Exception as e:
            error_logger(f"Could not estimate processing time for {path}: {e}")
            return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def build_graph_from_path_async(
        self, path: Path, is_dependency: bool = False, job_id: str | None = None
    ) -> None:
        try:
            if job_id:
                self.job_manager.update_job(job_id, status=JobStatus.RUNNING)

            await self.add_repository_to_graph(path, is_dependency)
            repo_name = path.name

            # Search for .cgcignore upwards
            curr = path.resolve()
            if not curr.is_dir():
                curr = curr.parent

            spec = None
            ignore_root = path.resolve()
            while True:
                candidate = curr / ".cgcignore"
                if candidate.exists():
                    ignore_root = curr
                    debug_log(f"Found .cgcignore at {ignore_root}")
                    with open(candidate) as f:
                        ignore_patterns = f.read().splitlines()
                    spec = pathspec.PathSpec.from_lines('gitwildmatch', ignore_patterns)
                    break
                if curr.parent == curr:
                    break
                curr = curr.parent

            supported_extensions = self.parsers.keys()
            all_files = path.rglob("*") if path.is_dir() else [path]
            files = [f for f in all_files if f.is_file() and f.suffix in supported_extensions]

            # Filter ignored directories
            ignore_dirs_str = get_config_value("IGNORE_DIRS") or ""
            if ignore_dirs_str and path.is_dir():
                ignore_dirs = {d.strip().lower() for d in ignore_dirs_str.split(',') if d.strip()}
                if ignore_dirs:
                    kept_files = []
                    for f in files:
                        try:
                            parts = set(p.lower() for p in f.relative_to(path).parent.parts)
                            if not parts.intersection(ignore_dirs):
                                kept_files.append(f)
                        except ValueError:
                            kept_files.append(f)
                    files = kept_files

            if spec:
                filtered_files = []
                for f in files:
                    try:
                        rel_path = f.relative_to(ignore_root)
                        if not spec.match_file(str(rel_path)):
                            filtered_files.append(f)
                        else:
                            debug_log(f"Ignored file based on .cgcignore: {rel_path}")
                    except ValueError:
                        filtered_files.append(f)
                files = filtered_files

            if job_id:
                self.job_manager.update_job(job_id, total_files=len(files))

            debug_log("Starting pre-scan to build imports map...")
            imports_map = self._pre_scan_for_imports(files)
            debug_log(f"Pre-scan complete. Found {len(imports_map)} definitions.")

            all_file_data: list[Dict] = []
            processed_count = 0

            for file in files:
                if file.is_file():
                    if job_id:
                        self.job_manager.update_job(job_id, current_file=str(file))
                    repo_path = path.resolve() if path.is_dir() else file.parent.resolve()
                    file_data = self.parse_file(repo_path, file, is_dependency)
                    if "error" not in file_data:
                        await self.add_file_to_graph(file_data, repo_name, imports_map)
                        all_file_data.append(file_data)
                    processed_count += 1
                    if job_id:
                        self.job_manager.update_job(job_id, processed_files=processed_count)
                    await asyncio.sleep(0.01)

            await self._create_all_inheritance_links(all_file_data, imports_map)
            await self._create_all_function_calls(all_file_data, imports_map)
            await self._embed_nodes(repo_name, job_id)

            if job_id:
                self.job_manager.update_job(
                    job_id, status=JobStatus.COMPLETED, end_time=datetime.now()
                )
        except Exception as e:
            error_message = str(e)
            error_logger(f"Failed to build graph for path {path}: {error_message}")
            if job_id:
                status = (
                    JobStatus.CANCELLED
                    if any(kw in error_message for kw in ("no such file found", "deleted", "not found"))
                    else JobStatus.FAILED
                )
                self.job_manager.update_job(
                    job_id, status=status, end_time=datetime.now(), errors=[error_message]
                )
