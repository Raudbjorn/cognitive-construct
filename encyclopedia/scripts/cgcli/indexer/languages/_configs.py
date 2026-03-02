"""Language configs for Python, JavaScript, TypeScript, TypeScriptJSX."""

from __future__ import annotations

from ._base import LanguageConfig, make_pre_scan
from . import _hooks as H


# =============================================================================
# Python
# =============================================================================

PY_QUERIES: dict[str, str] = {
    "imports": """
        (import_statement name: (_) @import)
        (import_from_statement) @from_import_stmt
    """,
    "classes": """
        (class_definition
            name: (identifier) @name
            superclasses: (argument_list)? @superclasses
            body: (block) @body)
    """,
    "functions": """
        (function_definition
            name: (identifier) @name
            parameters: (parameters) @parameters
            body: (block) @body
            return_type: (_)? @return_type)
    """,
    "calls": """
        (call
            function: (identifier) @name)
        (call
            function: (attribute attribute: (identifier) @name) @full_call)
    """,
    "variables": """
        (assignment
            left: (identifier) @name)
    """,
    "lambda_assignments": """
        (assignment
            left: (identifier) @name
            right: (lambda) @lambda_node)
    """,
    "docstrings": """
        (expression_statement (string) @docstring)
    """,
    "dict_method_refs": """
        (dictionary
            (pair
                key: (_) @key
                value: (attribute) @method_ref))
    """,
}

PYTHON_CONFIG = LanguageConfig(
    name="python",
    queries=PY_QUERIES,
    context_types=(
        "function_definition",
        "class_definition",
    ),
    complexity_nodes=frozenset({
        "if_statement",
        "for_statement",
        "while_statement",
        "except_clause",
        "with_statement",
        "boolean_operator",
        "list_comprehension",
        "generator_expression",
        "case_clause",
    }),
    docstring_strategy="first_string",
    find_functions=H.find_python_functions,
    find_classes=H.find_python_classes,
    find_imports=H.find_python_imports,
    find_calls=H.find_python_calls,
    find_variables=H.find_python_variables,
    find_extra={},
    pre_parse_hook=H.pre_parse_python_notebook,
    pre_scan_fn=H.pre_scan_python,
)


# =============================================================================
# JavaScript
# =============================================================================

JS_QUERIES: dict[str, str] = {
    "functions": """
        (function_declaration
            name: (identifier) @name
            parameters: (formal_parameters) @params
        ) @function_node

        (variable_declarator
            name: (identifier) @name
            value: (function_expression
                parameters: (formal_parameters) @params
            ) @function_node
        )

        (variable_declarator
            name: (identifier) @name
            value: (arrow_function
                parameters: (formal_parameters) @params
            ) @function_node
        )

        (variable_declarator
            name: (identifier) @name
            value: (arrow_function
                parameter: (identifier) @single_param
            ) @function_node
        )

        (method_definition
            name: (property_identifier) @name
            parameters: (formal_parameters) @params
        ) @function_node

        (assignment_expression
            left: (member_expression
                property: (property_identifier) @name
            )
            right: (function_expression
                parameters: (formal_parameters) @params
            ) @function_node
        )

        (assignment_expression
            left: (member_expression
                property: (property_identifier) @name
            )
            right: (arrow_function
                parameters: (formal_parameters) @params
            ) @function_node
        )
    """,
    "classes": """
        (class_declaration) @class
        (class) @class
    """,
    "imports": """
        (import_statement) @import
        (call_expression
            function: (identifier) @require_call (#eq? @require_call "require")
        ) @import
    """,
    "calls": """
        (call_expression function: (identifier) @name)
        (call_expression function: (member_expression property: (property_identifier) @name))
        (new_expression constructor: (identifier) @name)
        (new_expression constructor: (member_expression property: (property_identifier) @name))
    """,
    "variables": """
        (variable_declarator name: (identifier) @name)
    """,
    "docstrings": """
        (comment) @docstring_comment
    """,
}

JAVASCRIPT_CONFIG = LanguageConfig(
    name="javascript",
    queries=JS_QUERIES,
    context_types=(
        "function_declaration",
        "class_declaration",
        "function_expression",
        "method_definition",
        "arrow_function",
    ),
    complexity_nodes=frozenset({
        "if_statement",
        "for_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
        "case_statement",
        "conditional_expression",
        "logical_expression",
        "binary_expression",
        "catch_clause",
    }),
    docstring_strategy="jsdoc",
    find_functions=H.find_js_functions,
    find_classes=H.find_js_classes,
    find_imports=H.find_js_imports,
    find_calls=H.find_js_calls,
    find_variables=H.find_js_variables,
    find_extra={},
    pre_scan_fn=H.pre_scan_javascript,
)


# =============================================================================
# TypeScript
# =============================================================================

TS_QUERIES: dict[str, str] = {
    "functions": """
        (function_declaration
            name: (identifier) @name
            parameters: (formal_parameters) @params
        ) @function_node

        (variable_declarator
            name: (identifier) @name
            value: (function_expression
                parameters: (formal_parameters) @params
            ) @function_node
        )

        (variable_declarator
            name: (identifier) @name
            value: (arrow_function
                parameters: (formal_parameters) @params
            ) @function_node
        )

        (variable_declarator
            name: (identifier) @name
            value: (arrow_function
                parameter: (identifier) @single_param
            ) @function_node
        )

        (method_definition
            name: (property_identifier) @name
            parameters: (formal_parameters) @params
        ) @function_node

        (assignment_expression
            left: (member_expression
                property: (property_identifier) @name
            )
            right: (function_expression
                parameters: (formal_parameters) @params
            ) @function_node
        )

        (assignment_expression
            left: (member_expression
                property: (property_identifier) @name
            )
            right: (arrow_function
                parameters: (formal_parameters) @params
            ) @function_node
        )
    """,
    "classes": """
        (class_declaration) @class
        (abstract_class_declaration) @class
        (class) @class
    """,
    "interfaces": """
        (interface_declaration
            name: (type_identifier) @name
        ) @interface_node
    """,
    "type_aliases": """
        (type_alias_declaration
            name: (type_identifier) @name
        ) @type_alias_node
    """,
    "imports": """
        (import_statement) @import
        (call_expression
            function: (identifier) @require_call (#eq? @require_call "require")
        ) @import
    """,
    "calls": """
        (call_expression function: (identifier) @name)
        (call_expression function: (member_expression property: (property_identifier) @name))
        (new_expression constructor: (identifier) @name)
        (new_expression constructor: (member_expression property: (property_identifier) @name))
    """,
    "variables": """
        (variable_declarator name: (identifier) @name)
    """,
    "docstrings": """
        (comment) @docstring_comment
    """,
}

TYPESCRIPT_CONFIG = LanguageConfig(
    name="typescript",
    queries=TS_QUERIES,
    context_types=(
        "function_declaration",
        "class_declaration",
        "method_definition",
        "function_expression",
        "arrow_function",
    ),
    complexity_nodes=frozenset({
        "if_statement",
        "for_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
        "case_statement",
        "conditional_expression",
        "logical_expression",
        "binary_expression",
        "catch_clause",
    }),
    docstring_strategy="jsdoc",
    find_functions=H.find_ts_functions,
    find_classes=H.find_ts_classes,
    find_imports=H.find_ts_imports,
    find_calls=H.find_ts_calls,
    find_variables=H.find_ts_variables,
    find_extra={
        "interfaces": H.find_ts_interfaces,
        "type_aliases": H.find_ts_type_aliases,
    },
    pre_scan_fn=H.pre_scan_typescript,
)


# =============================================================================
# TypeScript JSX (.tsx)
# =============================================================================
# TSX inherits all TS queries — the tree-sitter tsx grammar is a superset of
# the TypeScript grammar so the same query strings work.  The only addition
# is the React component finder in find_extra.

TSX_QUERIES: dict[str, str] = TS_QUERIES  # same queries, different grammar

TYPESCRIPTJSX_CONFIG = LanguageConfig(
    name="typescript",  # language_name is "typescript" (matches original parser)
    queries=TSX_QUERIES,
    context_types=(
        "function_declaration",
        "class_declaration",
        "method_definition",
        "function_expression",
        "arrow_function",
    ),
    complexity_nodes=frozenset({
        "if_statement",
        "for_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
        "case_statement",
        "conditional_expression",
        "logical_expression",
        "binary_expression",
        "catch_clause",
    }),
    docstring_strategy="jsdoc",
    find_functions=H.find_ts_functions,
    find_classes=H.find_ts_classes,
    find_imports=H.find_ts_imports,
    find_calls=H.find_ts_calls,
    find_variables=H.find_ts_variables,
    find_extra={
        "interfaces": H.find_ts_interfaces,
        "type_aliases": H.find_ts_type_aliases,
        "components": H.find_tsx_react_components,
    },
    pre_scan_fn=H.pre_scan_typescriptjsx,
)


# =============================================================================
# Go
# =============================================================================

GO_QUERIES: dict[str, str] = {
    "functions": """
        (function_declaration
            name: (identifier) @name
            parameters: (parameter_list) @params
        ) @function_node

        (method_declaration
            receiver: (parameter_list) @receiver
            name: (field_identifier) @name
            parameters: (parameter_list) @params
        ) @function_node
    """,
    "structs": """
        (type_declaration
            (type_spec
                name: (type_identifier) @name
                type: (struct_type) @struct_body
            )
        ) @struct_node
    """,
    "interfaces": """
        (type_declaration
            (type_spec
                name: (type_identifier) @name
                type: (interface_type) @interface_body
            )
        ) @interface_node
    """,
    "imports": """
        (import_declaration
            (import_spec
                path: (interpreted_string_literal) @path
            )
        ) @import

        (import_declaration
            (import_spec
                name: (package_identifier) @alias
                path: (interpreted_string_literal) @path
            )
        ) @import_alias
    """,
    "calls": """
        (call_expression
            function: (identifier) @name
        )
        (call_expression
            function: (selector_expression
                field: (field_identifier) @name
            )
        )
    """,
    "variables": """
        (var_declaration
            (var_spec
                name: (identifier) @name
            )
        )
        (short_var_declaration
            left: (expression_list
                (identifier) @name
            )
        )
    """,
}

GO_CONFIG = LanguageConfig(
    name="go",
    queries=GO_QUERIES,
    context_types=(
        "function_declaration",
        "method_declaration",
        "type_declaration",
    ),
    complexity_nodes=frozenset({
        "if_statement",
        "for_statement",
        "switch_statement",
        "case_clause",
        "expression_switch_statement",
        "type_switch_statement",
        "binary_expression",
        "call_expression",
    }),
    docstring_strategy="prev_comment",
    find_functions=H.find_go_functions,
    find_classes=H.find_go_classes,
    find_imports=H.find_go_imports,
    find_calls=H.find_go_calls,
    find_variables=H.find_go_variables,
    find_extra={"interfaces": H.find_go_interfaces},
    pre_scan_fn=H.pre_scan_go,
)


# =============================================================================
# Rust
# =============================================================================

RUST_QUERIES: dict[str, str] = {
    "functions": """
        (function_item
            name: (identifier) @name
            parameters: (parameters) @params
        ) @function_node
    """,
    "classes": """
        [
            (struct_item name: (type_identifier) @name)
            (enum_item name: (type_identifier) @name)
            (trait_item name: (type_identifier) @name)
        ] @class
    """,
    "imports": """
        (use_declaration) @import
    """,
    "calls": """
        (call_expression
            function: [
                (identifier) @name
                (field_expression field: (field_identifier) @name)
                (scoped_identifier name: (identifier) @name)
            ]
        )
    """,
    "traits": """
        (trait_item name: (type_identifier) @name) @trait_node
    """,
}

RUST_CONFIG = LanguageConfig(
    name="rust",
    queries=RUST_QUERIES,
    context_types=(
        "function_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "impl_item",
    ),
    complexity_nodes=None,
    docstring_strategy="prev_comment",
    find_functions=H.find_rust_functions,
    find_classes=H.find_rust_classes,
    find_imports=H.find_rust_imports,
    find_calls=H.find_rust_calls,
    find_extra={"traits": H.find_rust_traits},
    pre_scan_fn=H.pre_scan_rust,
)


# =============================================================================
# Ruby
# =============================================================================

RUBY_QUERIES: dict[str, str] = {
    "functions": """
        (method
            name: (identifier) @name
        ) @function_node
    """,
    "classes": """
        (class
            name: (constant) @name
        ) @class
    """,
    "modules": """
        (module
            name: (constant) @name
        ) @module_node
    """,
    "imports": """
        (call
            method: (identifier) @method_name
            arguments: (argument_list
                (string) @path
            )
        ) @import
    """,
    "calls": """
        (call
            receiver: (_)? @receiver
            method: (identifier) @name
            arguments: (argument_list)? @args
        ) @call_node
    """,
    "variables": """
        (assignment
            left: (identifier) @name
            right: (_) @value
        )
        (assignment
            left: (instance_variable) @name
            right: (_) @value
        )
    """,
    "comments": """
        (comment) @comment
    """,
    "module_includes": """
        (call
          method: (identifier) @method
          arguments: (argument_list (constant) @module)
        ) @include_call
    """,
}

RUBY_CONFIG = LanguageConfig(
    name="ruby",
    queries=RUBY_QUERIES,
    context_types=(
        "class",
        "module",
        "method",
    ),
    complexity_nodes=frozenset({
        "if",
        "unless",
        "case",
        "when",
        "while",
        "until",
        "for",
        "rescue",
        "ensure",
        "and",
        "or",
        "&&",
        "||",
        "?",
        "ternary",
    }),
    docstring_strategy="prev_comment",
    find_functions=H.find_ruby_functions,
    find_classes=H.find_ruby_classes,
    find_imports=H.find_ruby_imports,
    find_calls=H.find_ruby_calls,
    find_variables=H.find_ruby_variables,
    find_extra={
        "modules": H.find_ruby_modules,
        "module_inclusions": H.find_ruby_module_inclusions,
    },
    pre_scan_fn=H.pre_scan_ruby,
)


# =============================================================================
# Java
# =============================================================================

JAVA_QUERIES: dict[str, str] = {
    "functions": """
        (method_declaration
            name: (identifier) @name
            parameters: (formal_parameters) @params
        ) @function_node

        (constructor_declaration
            name: (identifier) @name
            parameters: (formal_parameters) @params
        ) @function_node
    """,
    "classes": """
        [
            (class_declaration name: (identifier) @name)
            (interface_declaration name: (identifier) @name)
            (enum_declaration name: (identifier) @name)
            (annotation_type_declaration name: (identifier) @name)
        ] @class
    """,
    "imports": """
        (import_declaration) @import
    """,
    "calls": """
        (method_invocation
            name: (identifier) @name
        ) @call_node

        (object_creation_expression
            type: [
                (type_identifier)
                (scoped_type_identifier)
                (generic_type)
            ] @name
        ) @call_node
    """,
    "variables": """
        (local_variable_declaration
            type: (_) @type
            declarator: (variable_declarator
                name: (identifier) @name
            )
        ) @variable

        (field_declaration
            type: (_) @type
            declarator: (variable_declarator
                name: (identifier) @name
            )
        ) @variable
    """,
}

JAVA_CONFIG = LanguageConfig(
    name="java",
    queries=JAVA_QUERIES,
    context_types=(
        "method_declaration",
        "constructor_declaration",
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "annotation_type_declaration",
    ),
    complexity_nodes=None,
    docstring_strategy="none",
    find_functions=H.find_java_functions,
    find_classes=H.find_java_classes,
    find_imports=H.find_java_imports,
    find_calls=H.find_java_calls,
    find_variables=H.find_java_variables,
    find_extra={},
    pre_scan_fn=H.pre_scan_java,
)


# =============================================================================
# C
# =============================================================================

C_QUERIES: dict[str, str] = {
    "functions": """
        (function_definition
            declarator: (function_declarator
                declarator: (identifier) @name
            )
        ) @function_node

        (function_definition
            declarator: (function_declarator
                declarator: (pointer_declarator
                    declarator: (identifier) @name
                )
            )
        ) @function_node
    """,
    "structs": """
        (struct_specifier
            name: (type_identifier) @name
        ) @struct
    """,
    "unions": """
        (union_specifier
            name: (type_identifier) @name
        ) @union
    """,
    "enums": """
        (enum_specifier
            name: (type_identifier) @name
        ) @enum
    """,
    "typedefs": """
        (type_definition
            declarator: (type_identifier) @name
        ) @typedef
    """,
    "imports": """
        (preproc_include
            path: [
                (string_literal) @path
                (system_lib_string) @path
            ]
        ) @import
    """,
    "calls": """
        (call_expression
            function: (identifier) @name
        )
    """,
    "variables": """
        (declaration
            declarator: (init_declarator
                declarator: (identifier) @name
            )
        )

        (declaration
            declarator: (init_declarator
                declarator: (pointer_declarator
                    declarator: (identifier) @name
                )
            )
        )

        (declaration
            declarator: (identifier) @name
        )

        (declaration
            declarator: (pointer_declarator
                declarator: (identifier) @name
            )
        )
    """,
    "macros": """
        (preproc_def
            name: (identifier) @name
        ) @macro
    """,
}

C_CONFIG = LanguageConfig(
    name="c",
    queries=C_QUERIES,
    context_types=(
        "function_definition",
        "struct_specifier",
        "union_specifier",
        "enum_specifier",
    ),
    complexity_nodes=frozenset({
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
    }),
    docstring_strategy="prev_comment",
    find_functions=H.find_c_functions,
    find_classes=H.find_c_classes,
    find_imports=H.find_c_imports,
    find_calls=H.find_c_calls,
    find_variables=H.find_c_variables,
    find_extra={
        "macros": H.find_c_macros,
    },
    pre_scan_fn=H.pre_scan_c,
)


# =============================================================================
# C++
# =============================================================================

CPP_QUERIES: dict[str, str] = {
    "functions": """
        (function_definition
            declarator: (function_declarator
                declarator: [
                    (identifier) @name
                    (field_identifier) @name
                ]
            )
        ) @function_node
    """,
    "classes": """
        (class_specifier
            name: (type_identifier) @name
        ) @class
    """,
    "imports": """
        (preproc_include
            path: [
                (string_literal) @path
                (system_lib_string) @path
            ]
        ) @import
    """,
    "calls": """
        (call_expression
            function: [
                (identifier) @function_name
                (field_expression
                    field: (field_identifier) @method_name
                )
            ]
        arguments: (argument_list) @args
    )
    """,
    "enums":"""
        (enum_specifier
            name: (type_identifier) @name
            body: (enumerator_list
                (enumerator
                    name: (identifier) @value
                    )*
                )? @body
        ) @enum
    """,
    "structs":"""
        (struct_specifier
            name: (type_identifier) @name
            body: (field_declaration_list)? @body
        ) @struct
    """,
    "unions": """
    (union_specifier
    name: (type_identifier)? @name
    body: (field_declaration_list
        (field_declaration
            declarator: [
                (field_identifier) @value
                (pointer_declarator (field_identifier) @value)
                (array_declarator (field_identifier) @value)
                ]
            )*
        )? @body
    ) @union

    """,
    "macros": """
        (preproc_def
            name: (identifier) @name
        ) @macro
    """,
    "variables": """
    (declaration
        declarator: (init_declarator
                        declarator: (identifier) @name))

    (declaration
        declarator: (init_declarator
                        declarator: (pointer_declarator
                            declarator: (identifier) @name)))

    (field_declaration
        declarator: [
             (field_identifier) @name
             (pointer_declarator declarator: (field_identifier) @name)
             (array_declarator declarator: (field_identifier) @name)
             (reference_declarator (field_identifier) @name)
        ]
    )
    """,
    "lambda_assignments": """
    ; Match a lambda assigned to a variable
    (declaration
        declarator: (init_declarator
            declarator: (identifier) @name
            value: (lambda_expression) @lambda_node))
    """

}

CPP_CONFIG = LanguageConfig(
    name="cpp",
    queries=CPP_QUERIES,
    context_types=(
        "function_definition",
        "class_specifier",
        "namespace_definition",
    ),
    complexity_nodes=frozenset({
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
        "catch_clause",
    }),
    docstring_strategy="prev_comment",
    find_functions=H.find_cpp_functions,
    find_classes=H.find_cpp_classes,
    find_imports=H.find_cpp_imports,
    find_calls=H.find_cpp_calls,
    find_variables=H.find_cpp_variables,
    find_extra={
        "structs": H.find_cpp_structs,
        "enums": H.find_cpp_enums,
        "unions": H.find_cpp_unions,
        "macros": H.find_cpp_macros,
        "lambda_assignments": H.find_cpp_lambda_assignments,
    },
    pre_scan_fn=H.pre_scan_cpp,
)


# =============================================================================
# C#
# =============================================================================

CSHARP_QUERIES: dict[str, str] = {
    "functions": """
        (method_declaration
            name: (identifier) @name
            parameters: (parameter_list) @params
        ) @function_node

        (constructor_declaration
            name: (identifier) @name
            parameters: (parameter_list) @params
        ) @function_node

        (local_function_statement
            name: (identifier) @name
            parameters: (parameter_list) @params
        ) @function_node
    """,
    "classes": """
        (class_declaration
            name: (identifier) @name
            (base_list)? @bases
        ) @class
    """,
    "interfaces": """
        (interface_declaration
            name: (identifier) @name
            (base_list)? @bases
        ) @interface
    """,
    "structs": """
        (struct_declaration
            name: (identifier) @name
            (base_list)? @bases
        ) @struct
    """,
    "enums": """
        (enum_declaration
            name: (identifier) @name
        ) @enum
    """,
    "records": """
        (record_declaration
            name: (identifier) @name
            (base_list)? @bases
        ) @record
    """,
    "properties": """
        (property_declaration
            name: (identifier) @name
        ) @property
    """,
    "imports": """
        (using_directive) @import
    """,
    "calls": """
        (invocation_expression
            function: [
                (identifier) @name
                (member_access_expression
                    name: (identifier) @name
                )
            ]
        )

        (object_creation_expression
            type: [
                (identifier) @name
                (qualified_name) @name
            ]
        )
    """,
}

CSHARP_CONFIG = LanguageConfig(
    name="c_sharp",
    queries=CSHARP_QUERIES,
    context_types=(
        "class_declaration",
        "struct_declaration",
        "interface_declaration",
        "record_declaration",
        "method_declaration",
        "constructor_declaration",
    ),
    complexity_nodes=frozenset({
        "if_statement",
        "for_statement",
        "for_each_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
        "case_switch_label",
        "conditional_expression",
        "binary_expression",
        "catch_clause",
    }),
    docstring_strategy="prev_comment",
    find_functions=H.find_csharp_functions,
    find_classes=H.find_csharp_classes,
    find_imports=H.find_csharp_imports,
    find_calls=H.find_csharp_calls,
    find_extra={
        "interfaces": H.find_csharp_interfaces,
        "structs": H.find_csharp_structs,
        "enums": H.find_csharp_enums,
        "records": H.find_csharp_records,
        "properties": H.find_csharp_properties,
    },
    pre_scan_fn=H.pre_scan_csharp,
)


# =============================================================================
# PHP
# =============================================================================

PHP_QUERIES: dict[str, str] = {
    "functions": """
        (function_definition
            name: (name) @name
            parameters: (formal_parameters) @params
        ) @function_node

        (method_declaration
            name: (name) @name
            parameters: (formal_parameters) @params
        ) @function_node
    """,
    "classes": """
        (class_declaration
            name: (name) @name
        ) @class

        (interface_declaration
            name: (name) @name
        ) @interface

        (trait_declaration
            name: (name) @name
        ) @trait
    """,
    "imports": """
        (use_declaration) @import
    """,
    "calls": """
        (function_call_expression
            function: [
                (qualified_name) @name
                (name) @name
            ]
        ) @call_node

        (member_call_expression
            name: (name) @name
        ) @call_node

        (scoped_call_expression
            name: (name) @name
        ) @call_node

        (object_creation_expression) @call_node
    """,
    "variables": """
        (variable_name) @variable
    """,
}

PHP_CONFIG = LanguageConfig(
    name="php",
    queries=PHP_QUERIES,
    context_types=(
        "function_definition",
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "trait_declaration",
    ),
    complexity_nodes=frozenset({
        "if_statement",
        "for_statement",
        "foreach_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
        "case_statement",
        "conditional_expression",
        "binary_expression",
        "catch_clause",
    }),
    docstring_strategy="prev_comment",
    find_functions=H.find_php_functions,
    find_classes=H.find_php_classes,
    find_imports=H.find_php_imports,
    find_calls=H.find_php_calls,
    find_variables=H.find_php_variables,
    find_extra={
        "interfaces": H.find_php_interfaces,
        "traits": H.find_php_traits,
    },
    pre_scan_fn=H.pre_scan_php,
)


# =============================================================================
# Swift
# =============================================================================

SWIFT_QUERIES: dict[str, str] = {
    "functions": """
        [
            (function_declaration
                name: (simple_identifier) @name
                parameters: (parameter)* @params
            ) @function_node
            (init_declaration
                parameters: (parameter)* @params
            ) @init_node
        ]
    """,
    "classes": """
        [
            (class_declaration
                name: (type_identifier) @name
            ) @class
            (struct_declaration
                name: (type_identifier) @name
            ) @struct
            (enum_declaration
                name: (type_identifier) @name
            ) @enum
            (protocol_declaration
                name: (type_identifier) @name
            ) @protocol
        ]
    """,
    "imports": """
        (import_declaration) @import
    """,
    "calls": """
        (call_expression) @call_node
    """,
    "variables": """
        [
            (property_declaration
                (pattern) @pattern
            ) @variable
            (constant_declaration
                (pattern_binding
                    pattern: (simple_identifier) @name
                )
            ) @constant
        ]
    """,
}

SWIFT_CONFIG = LanguageConfig(
    name="swift",
    queries=SWIFT_QUERIES,
    context_types=(
        "function_declaration",
        "init_declaration",
        "class_declaration",
        "struct_declaration",
        "enum_declaration",
        "protocol_declaration",
    ),
    docstring_strategy="prev_comment",
    find_functions=H.find_swift_functions,
    find_classes=H.find_swift_classes,
    find_imports=H.find_swift_imports,
    find_calls=H.find_swift_calls,
    find_variables=H.find_swift_variables,
    find_extra={
        "structs": H.find_swift_structs,
        "enums": H.find_swift_enums,
        "protocols": H.find_swift_protocols,
    },
    pre_scan_fn=H.pre_scan_swift,
)


# =============================================================================
# Kotlin
# =============================================================================

KOTLIN_QUERIES: dict[str, str] = {
    "functions": """
        (function_declaration
            (simple_identifier) @name
            (function_value_parameters) @params
        ) @function_node
    """,
    "classes": """
        [
            (class_declaration (type_identifier) @name)
            (object_declaration (type_identifier) @name)
            (companion_object (type_identifier)? @name)
        ] @class
    """,
    "imports": """
        (import_header) @import
    """,
    "calls": """
        (call_expression) @call_node
    """,
    "variables": """
        (property_declaration
            (variable_declaration
                (simple_identifier) @name
            )
        ) @variable
    """,
}

KOTLIN_CONFIG = LanguageConfig(
    name="kotlin",
    queries=KOTLIN_QUERIES,
    context_types=(
        "function_declaration",
        "class_declaration",
        "object_declaration",
        "companion_object",
        "object_literal",
    ),
    find_functions=H.find_kotlin_functions,
    find_classes=H.find_kotlin_classes,
    find_imports=H.find_kotlin_imports,
    find_calls=H.find_kotlin_calls,
    find_variables=H.find_kotlin_variables,
    pre_scan_fn=H.pre_scan_kotlin,
)


# =============================================================================
# Scala
# =============================================================================

SCALA_QUERIES: dict[str, str] = {
    "functions": """
        (function_definition
            name: (identifier) @name
            parameters: (parameters) @params
        ) @function_node
    """,
    "classes": """
        [
            (class_definition name: (identifier) @name)
            (object_definition name: (identifier) @name)
            (trait_definition name: (identifier) @name)
        ] @class
    """,
    "imports": """
        (import_declaration) @import
    """,
    "calls": """
        (call_expression) @call_node
        (generic_function
             function: (identifier) @name
        ) @call_node
    """,
    "variables": """
        (val_definition
            pattern: (identifier) @name
        ) @variable

        (var_definition
            pattern: (identifier) @name
        ) @variable
    """,
}

SCALA_CONFIG = LanguageConfig(
    name="scala",
    queries=SCALA_QUERIES,
    context_types=(
        "function_definition",
        "class_definition",
        "object_definition",
        "trait_definition",
    ),
    find_functions=H.find_scala_functions,
    find_classes=H.find_scala_classes,
    find_imports=H.find_scala_imports,
    find_calls=H.find_scala_calls,
    find_variables=H.find_scala_variables,
    find_extra={
        "traits": H.find_scala_traits,
    },
    pre_scan_fn=H.pre_scan_scala,
)


# =============================================================================
# Haskell
# =============================================================================
# NOTE: This config preserves the original buggy queries and behaviour.
# The queries use Swift/Kotlin node types (simple_identifier, type_identifier,
# function_declaration, class_declaration, struct_declaration, enum_declaration,
# protocol_declaration, property_declaration, variable_declaration,
# call_expression, import_declaration, navigation_expression, etc.) which are
# WRONG for the Haskell tree-sitter grammar.  The pre_scan_haskell function
# searches for "package" and "class|object|interface|typealias" keywords which
# also do not exist in Haskell.  All of this is kept intentionally to match
# the original parser file exactly.

HASKELL_QUERIES: dict[str, str] = {
    "functions": """
        [
        (function_declaration
            name: (simple_identifier) @name
            parameters: (parameters)* @params
        ) @function_node
        (init_declaration
            parameters: (parameter)* @params
        ) @init_node
        ]
    """,
    "classes": """
    [
        (class_declaration
            name: (type_identifier) @name
        ) @class
        (
        struct_declaration
            name: (type_identifier) @name
        ) @struct
        (
            enum_declaration
            name: (type_identifier) @name
        ) @enum
        (
            protocol_declaration
            name: (type_identifier) @name
        ) @protocol
    ]
    """,
    "imports": """
        (import_declaration) @import
    """,
    "calls": """
        (call_expression) @call_node
    """,
    "variables": """
        (property_declaration
            (variable_declaration
                (simple_identifier) @name
            )
        ) @variable
    """,
}

HASKELL_CONFIG = LanguageConfig(
    name="haskell",
    queries=HASKELL_QUERIES,
    context_types=(
        "class_declaration",
        "object_declaration",
        "companion_object",
        "object_literal",
    ),
    find_functions=H.find_haskell_functions,
    find_classes=H.find_haskell_classes,
    find_imports=H.find_haskell_imports,
    find_calls=H.find_haskell_calls,
    find_variables=H.find_haskell_variables,
    pre_scan_fn=H.pre_scan_haskell,
)



# ===========================================================================
# Config registry
# ===========================================================================

CONFIGS: dict[str, LanguageConfig] = {
    "python": PYTHON_CONFIG,
    "javascript": JAVASCRIPT_CONFIG,
    "typescript": TYPESCRIPT_CONFIG,
    "typescriptjsx": TYPESCRIPTJSX_CONFIG,
    "go": GO_CONFIG,
    "rust": RUST_CONFIG,
    "ruby": RUBY_CONFIG,
    "java": JAVA_CONFIG,
    "c": C_CONFIG,
    "cpp": CPP_CONFIG,
    "c_sharp": CSHARP_CONFIG,
    "php": PHP_CONFIG,
    "swift": SWIFT_CONFIG,
    "kotlin": KOTLIN_CONFIG,
    "scala": SCALA_CONFIG,
    "haskell": HASKELL_CONFIG,
}


def get_config(language_name: str) -> LanguageConfig:
    """Look up a LanguageConfig by language name.

    Raises KeyError if the language is not supported.
    """
    return CONFIGS[language_name]
