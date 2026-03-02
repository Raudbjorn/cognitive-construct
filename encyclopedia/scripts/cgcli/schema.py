"""SurrealQL schema definition for the code graph database.

Applied idempotently on every connection via DEFINE ... IF NOT EXISTS.
"""

SCHEMA = """
-- Analyzers
DEFINE ANALYZER IF NOT EXISTS code_analyzer
    TOKENIZERS class, blank
    FILTERS lowercase, snowball(english);

-- Core tables
DEFINE TABLE IF NOT EXISTS repository SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS name         ON repository TYPE string;
DEFINE FIELD IF NOT EXISTS path         ON repository TYPE string;
DEFINE FIELD IF NOT EXISTS is_dependency ON repository TYPE bool DEFAULT false;
DEFINE INDEX IF NOT EXISTS idx_repo_path ON repository FIELDS path UNIQUE;

DEFINE TABLE IF NOT EXISTS node SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS node_type     ON node TYPE string;
DEFINE FIELD IF NOT EXISTS name          ON node TYPE string;
DEFINE FIELD IF NOT EXISTS path          ON node TYPE option<string>;
DEFINE FIELD IF NOT EXISTS line_number   ON node TYPE option<int>;
DEFINE FIELD IF NOT EXISTS end_line      ON node TYPE option<int>;
DEFINE FIELD IF NOT EXISTS source        ON node TYPE option<string>;
DEFINE FIELD IF NOT EXISTS docstring     ON node TYPE option<string>;
DEFINE FIELD IF NOT EXISTS language      ON node TYPE option<string>;
DEFINE FIELD IF NOT EXISTS is_dependency ON node TYPE bool DEFAULT false;
DEFINE FIELD IF NOT EXISTS is_searchable ON node TYPE bool DEFAULT false;
DEFINE FIELD IF NOT EXISTS args          ON node TYPE option<array>;
DEFINE FIELD IF NOT EXISTS decorators    ON node TYPE option<array>;
DEFINE FIELD IF NOT EXISTS complexity    ON node TYPE option<int>;
DEFINE FIELD IF NOT EXISTS bases         ON node TYPE option<array>;
DEFINE FIELD IF NOT EXISTS value         ON node TYPE option<string>;
DEFINE FIELD IF NOT EXISTS context       ON node TYPE option<string>;
DEFINE FIELD IF NOT EXISTS alias         ON node TYPE option<string>;
DEFINE FIELD IF NOT EXISTS full_import_name ON node TYPE option<string>;
DEFINE FIELD IF NOT EXISTS relative_path ON node TYPE option<string>;
DEFINE FIELD IF NOT EXISTS repo          ON node TYPE option<record<repository>>;
DEFINE FIELD IF NOT EXISTS embedding     ON node TYPE option<array<float>>;

-- Node indexes
DEFINE INDEX IF NOT EXISTS idx_node_unique    ON node FIELDS node_type, name, path, line_number UNIQUE;
DEFINE INDEX IF NOT EXISTS idx_node_type      ON node FIELDS node_type;
DEFINE INDEX IF NOT EXISTS idx_node_name      ON node FIELDS name;
DEFINE INDEX IF NOT EXISTS idx_node_path      ON node FIELDS path;
DEFINE INDEX IF NOT EXISTS idx_node_language  ON node FIELDS language;
DEFINE INDEX IF NOT EXISTS idx_node_repo      ON node FIELDS repo;

-- Full-text search (BM25)
DEFINE INDEX IF NOT EXISTS idx_node_name_search ON node
    FIELDS name SEARCH ANALYZER code_analyzer BM25(1.2, 0.75);
DEFINE INDEX IF NOT EXISTS idx_node_source_search ON node
    FIELDS source SEARCH ANALYZER code_analyzer BM25(1.2, 0.75);
DEFINE INDEX IF NOT EXISTS idx_node_docstring_search ON node
    FIELDS docstring SEARCH ANALYZER code_analyzer BM25(1.2, 0.75);

-- Vector index (HNSW, 384-dim, cosine)
DEFINE INDEX IF NOT EXISTS idx_node_embedding ON node
    FIELDS embedding HNSW DIMENSION 384 DIST COSINE EFC 150 M 12;

-- Graph edge tables (native RELATION type)
DEFINE TABLE IF NOT EXISTS contains   TYPE RELATION IN node|repository OUT node;

DEFINE TABLE IF NOT EXISTS calls      TYPE RELATION IN node OUT node;
DEFINE FIELD IF NOT EXISTS line_number    ON calls TYPE option<int>;
DEFINE FIELD IF NOT EXISTS args           ON calls TYPE option<array>;
DEFINE FIELD IF NOT EXISTS full_call_name ON calls TYPE option<string>;

DEFINE TABLE IF NOT EXISTS inherits   TYPE RELATION IN node OUT node;
DEFINE TABLE IF NOT EXISTS implements TYPE RELATION IN node OUT node;

DEFINE TABLE IF NOT EXISTS imports    TYPE RELATION IN node OUT node;
DEFINE FIELD IF NOT EXISTS alias         ON imports TYPE option<string>;
DEFINE FIELD IF NOT EXISTS imported_name ON imports TYPE option<string>;
DEFINE FIELD IF NOT EXISTS line_number   ON imports TYPE option<int>;

DEFINE TABLE IF NOT EXISTS has_parameter TYPE RELATION IN node OUT node;
DEFINE TABLE IF NOT EXISTS includes      TYPE RELATION IN node OUT node;

-- Edge indexes
DEFINE INDEX IF NOT EXISTS idx_calls_in   ON calls FIELDS in;
DEFINE INDEX IF NOT EXISTS idx_calls_out  ON calls FIELDS out;
DEFINE INDEX IF NOT EXISTS idx_imports_in ON imports FIELDS in;
"""
