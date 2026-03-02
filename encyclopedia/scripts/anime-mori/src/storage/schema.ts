/**
 * SurrealQL schema for Anime Mori.
 *
 * All definitions use IF NOT EXISTS for idempotent application.
 * Applied once on SurrealStorage.create() — no migration system needed.
 */

export const SCHEMA_VERSION = 1;

export const SCHEMA_V1 = `
-- ── Analyzers ──────────────────────────────────────────────────────────

DEFINE ANALYZER IF NOT EXISTS code_analyzer
  TOKENIZERS blank
  FILTERS lowercase, ascii;

DEFINE ANALYZER IF NOT EXISTS concept_analyzer
  TOKENIZERS blank, class
  FILTERS lowercase, ascii;

-- ── Schema version tracking ────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS schema_version SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS version  ON schema_version TYPE int;
DEFINE FIELD IF NOT EXISTS applied  ON schema_version TYPE datetime DEFAULT time::now();

-- ── Semantic Concepts ──────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS semantic_concept SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS concept_name      ON semantic_concept TYPE string;
DEFINE FIELD IF NOT EXISTS concept_type      ON semantic_concept TYPE string;
DEFINE FIELD IF NOT EXISTS confidence_score  ON semantic_concept TYPE float  DEFAULT 0.0;
DEFINE FIELD IF NOT EXISTS relationships     ON semantic_concept TYPE object DEFAULT {};
DEFINE FIELD IF NOT EXISTS evolution_history ON semantic_concept TYPE object DEFAULT {};
DEFINE FIELD IF NOT EXISTS file_path         ON semantic_concept TYPE string;
DEFINE FIELD IF NOT EXISTS line_range        ON semantic_concept TYPE object DEFAULT { start: 0, end: 0 };
DEFINE FIELD IF NOT EXISTS created_at        ON semantic_concept TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at        ON semantic_concept TYPE datetime DEFAULT time::now();

DEFINE INDEX IF NOT EXISTS idx_sc_type     ON semantic_concept COLUMNS concept_type;
DEFINE INDEX IF NOT EXISTS idx_sc_file     ON semantic_concept COLUMNS file_path;
DEFINE INDEX IF NOT EXISTS idx_sc_name_ft  ON semantic_concept COLUMNS concept_name
  SEARCH ANALYZER concept_analyzer BM25(1.2, 0.75);

-- ── Developer Patterns ─────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS developer_pattern SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS pattern_type    ON developer_pattern TYPE string;
DEFINE FIELD IF NOT EXISTS pattern_content ON developer_pattern TYPE object;
DEFINE FIELD IF NOT EXISTS frequency       ON developer_pattern TYPE int     DEFAULT 1;
DEFINE FIELD IF NOT EXISTS contexts        ON developer_pattern TYPE array   DEFAULT [];
DEFINE FIELD IF NOT EXISTS contexts.*      ON developer_pattern TYPE string;
DEFINE FIELD IF NOT EXISTS examples        ON developer_pattern TYPE array   DEFAULT [];
DEFINE FIELD IF NOT EXISTS confidence      ON developer_pattern TYPE float   DEFAULT 0.0;
DEFINE FIELD IF NOT EXISTS created_at      ON developer_pattern TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS last_seen       ON developer_pattern TYPE datetime DEFAULT time::now();

DEFINE INDEX IF NOT EXISTS idx_dp_type ON developer_pattern COLUMNS pattern_type;
DEFINE INDEX IF NOT EXISTS idx_dp_freq ON developer_pattern COLUMNS frequency;

-- ── File Intelligence ──────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS file_intelligence SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS file_path           ON file_intelligence TYPE string;
DEFINE FIELD IF NOT EXISTS file_hash           ON file_intelligence TYPE string;
DEFINE FIELD IF NOT EXISTS semantic_concepts   ON file_intelligence TYPE array  DEFAULT [];
DEFINE FIELD IF NOT EXISTS semantic_concepts.* ON file_intelligence TYPE string;
DEFINE FIELD IF NOT EXISTS patterns_used       ON file_intelligence TYPE array  DEFAULT [];
DEFINE FIELD IF NOT EXISTS patterns_used.*     ON file_intelligence TYPE string;
DEFINE FIELD IF NOT EXISTS complexity_metrics  ON file_intelligence TYPE object DEFAULT {};
DEFINE FIELD IF NOT EXISTS dependencies        ON file_intelligence TYPE array  DEFAULT [];
DEFINE FIELD IF NOT EXISTS dependencies.*      ON file_intelligence TYPE string;
DEFINE FIELD IF NOT EXISTS last_analyzed       ON file_intelligence TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS created_at          ON file_intelligence TYPE datetime DEFAULT time::now();

DEFINE INDEX IF NOT EXISTS idx_fi_path     ON file_intelligence COLUMNS file_path UNIQUE;
DEFINE INDEX IF NOT EXISTS idx_fi_analyzed ON file_intelligence COLUMNS last_analyzed;

-- ── AI Insights ────────────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS ai_insight SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS insight_type      ON ai_insight TYPE string;
DEFINE FIELD IF NOT EXISTS insight_content   ON ai_insight TYPE object;
DEFINE FIELD IF NOT EXISTS confidence_score  ON ai_insight TYPE float   DEFAULT 0.0;
DEFINE FIELD IF NOT EXISTS source_agent      ON ai_insight TYPE string;
DEFINE FIELD IF NOT EXISTS validation_status ON ai_insight TYPE string  DEFAULT 'pending';
DEFINE FIELD IF NOT EXISTS impact_prediction ON ai_insight TYPE object  DEFAULT {};
DEFINE FIELD IF NOT EXISTS created_at        ON ai_insight TYPE datetime DEFAULT time::now();

DEFINE INDEX IF NOT EXISTS idx_ai_type       ON ai_insight COLUMNS insight_type;
DEFINE INDEX IF NOT EXISTS idx_ai_confidence ON ai_insight COLUMNS confidence_score;

-- ── Architectural Decisions ────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS architectural_decision SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS decision_context     ON architectural_decision TYPE string;
DEFINE FIELD IF NOT EXISTS decision_rationale   ON architectural_decision TYPE option<string>;
DEFINE FIELD IF NOT EXISTS alternatives_considered ON architectural_decision TYPE object DEFAULT {};
DEFINE FIELD IF NOT EXISTS impact_analysis      ON architectural_decision TYPE object DEFAULT {};
DEFINE FIELD IF NOT EXISTS decision_date        ON architectural_decision TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS files_affected       ON architectural_decision TYPE array   DEFAULT [];
DEFINE FIELD IF NOT EXISTS files_affected.*     ON architectural_decision TYPE string;
DEFINE FIELD IF NOT EXISTS created_at           ON architectural_decision TYPE datetime DEFAULT time::now();

-- ── Shared Patterns ───────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS shared_pattern SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS pattern_name        ON shared_pattern TYPE string;
DEFINE FIELD IF NOT EXISTS pattern_description ON shared_pattern TYPE option<string>;
DEFINE FIELD IF NOT EXISTS pattern_data        ON shared_pattern TYPE object DEFAULT {};
DEFINE FIELD IF NOT EXISTS usage_count         ON shared_pattern TYPE int    DEFAULT 0;
DEFINE FIELD IF NOT EXISTS community_rating    ON shared_pattern TYPE float  DEFAULT 0.0;
DEFINE FIELD IF NOT EXISTS tags                ON shared_pattern TYPE array  DEFAULT [];
DEFINE FIELD IF NOT EXISTS tags.*              ON shared_pattern TYPE string;
DEFINE FIELD IF NOT EXISTS created_at          ON shared_pattern TYPE datetime DEFAULT time::now();

-- ── Project Metadata ──────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS project_metadata SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS project_path         ON project_metadata TYPE string;
DEFINE FIELD IF NOT EXISTS project_name         ON project_metadata TYPE option<string>;
DEFINE FIELD IF NOT EXISTS language_primary     ON project_metadata TYPE option<string>;
DEFINE FIELD IF NOT EXISTS languages_detected   ON project_metadata TYPE array  DEFAULT [];
DEFINE FIELD IF NOT EXISTS languages_detected.* ON project_metadata TYPE string;
DEFINE FIELD IF NOT EXISTS framework_detected   ON project_metadata TYPE array  DEFAULT [];
DEFINE FIELD IF NOT EXISTS framework_detected.* ON project_metadata TYPE string;
DEFINE FIELD IF NOT EXISTS intelligence_version ON project_metadata TYPE option<string>;
DEFINE FIELD IF NOT EXISTS last_full_scan       ON project_metadata TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS created_at           ON project_metadata TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at           ON project_metadata TYPE datetime DEFAULT time::now();

DEFINE INDEX IF NOT EXISTS idx_pm_path ON project_metadata COLUMNS project_path UNIQUE;

-- ── Feature Map ───────────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS feature_map SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS project_path      ON feature_map TYPE string;
DEFINE FIELD IF NOT EXISTS feature_name      ON feature_map TYPE string;
DEFINE FIELD IF NOT EXISTS primary_files     ON feature_map TYPE array  DEFAULT [];
DEFINE FIELD IF NOT EXISTS primary_files.*   ON feature_map TYPE string;
DEFINE FIELD IF NOT EXISTS related_files     ON feature_map TYPE array  DEFAULT [];
DEFINE FIELD IF NOT EXISTS related_files.*   ON feature_map TYPE string;
DEFINE FIELD IF NOT EXISTS dependencies      ON feature_map TYPE array  DEFAULT [];
DEFINE FIELD IF NOT EXISTS dependencies.*    ON feature_map TYPE string;
DEFINE FIELD IF NOT EXISTS status            ON feature_map TYPE string DEFAULT 'active';
DEFINE FIELD IF NOT EXISTS created_at        ON feature_map TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at        ON feature_map TYPE datetime DEFAULT time::now();

DEFINE INDEX IF NOT EXISTS idx_fm_project ON feature_map COLUMNS project_path;
DEFINE INDEX IF NOT EXISTS idx_fm_name    ON feature_map COLUMNS feature_name;

-- ── Entry Points ──────────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS entry_point SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS project_path ON entry_point TYPE string;
DEFINE FIELD IF NOT EXISTS entry_type   ON entry_point TYPE string;
DEFINE FIELD IF NOT EXISTS file_path    ON entry_point TYPE string;
DEFINE FIELD IF NOT EXISTS description  ON entry_point TYPE option<string>;
DEFINE FIELD IF NOT EXISTS framework    ON entry_point TYPE option<string>;
DEFINE FIELD IF NOT EXISTS created_at   ON entry_point TYPE datetime DEFAULT time::now();

DEFINE INDEX IF NOT EXISTS idx_ep_project ON entry_point COLUMNS project_path;

-- ── Key Directories ───────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS key_directory SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS project_path   ON key_directory TYPE string;
DEFINE FIELD IF NOT EXISTS directory_path ON key_directory TYPE string;
DEFINE FIELD IF NOT EXISTS directory_type ON key_directory TYPE string;
DEFINE FIELD IF NOT EXISTS file_count     ON key_directory TYPE int    DEFAULT 0;
DEFINE FIELD IF NOT EXISTS description    ON key_directory TYPE option<string>;
DEFINE FIELD IF NOT EXISTS created_at     ON key_directory TYPE datetime DEFAULT time::now();

DEFINE INDEX IF NOT EXISTS idx_kd_project ON key_directory COLUMNS project_path;

-- ── Work Sessions ─────────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS work_session SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS project_path     ON work_session TYPE string;
DEFINE FIELD IF NOT EXISTS session_start    ON work_session TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS session_end      ON work_session TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS last_feature     ON work_session TYPE option<string>;
DEFINE FIELD IF NOT EXISTS current_files    ON work_session TYPE array  DEFAULT [];
DEFINE FIELD IF NOT EXISTS current_files.*  ON work_session TYPE string;
DEFINE FIELD IF NOT EXISTS completed_tasks  ON work_session TYPE array  DEFAULT [];
DEFINE FIELD IF NOT EXISTS completed_tasks.* ON work_session TYPE string;
DEFINE FIELD IF NOT EXISTS pending_tasks    ON work_session TYPE array  DEFAULT [];
DEFINE FIELD IF NOT EXISTS pending_tasks.*  ON work_session TYPE string;
DEFINE FIELD IF NOT EXISTS blockers         ON work_session TYPE array  DEFAULT [];
DEFINE FIELD IF NOT EXISTS blockers.*       ON work_session TYPE string;
DEFINE FIELD IF NOT EXISTS session_notes    ON work_session TYPE option<string>;
DEFINE FIELD IF NOT EXISTS last_updated     ON work_session TYPE datetime DEFAULT time::now();

DEFINE INDEX IF NOT EXISTS idx_ws_project ON work_session COLUMNS project_path;
DEFINE INDEX IF NOT EXISTS idx_ws_updated ON work_session COLUMNS last_updated;

-- ── Project Decisions ─────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS project_decision SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS project_path   ON project_decision TYPE string;
DEFINE FIELD IF NOT EXISTS decision_key   ON project_decision TYPE string;
DEFINE FIELD IF NOT EXISTS decision_value ON project_decision TYPE string;
DEFINE FIELD IF NOT EXISTS reasoning      ON project_decision TYPE option<string>;
DEFINE FIELD IF NOT EXISTS made_at        ON project_decision TYPE datetime DEFAULT time::now();

DEFINE INDEX IF NOT EXISTS idx_pd_path_key ON project_decision COLUMNS project_path, decision_key UNIQUE;

-- ── Code Documents (embeddings) ───────────────────────────────────────

DEFINE TABLE IF NOT EXISTS code_document SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS code          ON code_document TYPE string;
DEFINE FIELD IF NOT EXISTS embedding     ON code_document TYPE array  DEFAULT [];
DEFINE FIELD IF NOT EXISTS metadata      ON code_document TYPE object DEFAULT {};
DEFINE FIELD IF NOT EXISTS created_at    ON code_document TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at    ON code_document TYPE datetime DEFAULT time::now();

DEFINE INDEX IF NOT EXISTS idx_cd_code_ft ON code_document COLUMNS code
  SEARCH ANALYZER code_analyzer BM25(1.2, 0.75) HIGHLIGHTS;

DEFINE INDEX IF NOT EXISTS idx_cd_embedding ON code_document COLUMNS embedding
  MTREE DIMENSION 384 DIST COSINE TYPE F32;
`;
