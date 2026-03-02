/**
 * Unified SurrealDB storage layer for Anime Mori.
 * Replaces both SQLiteDatabase and SemanticVectorDB with a single async class.
 */

import { Surreal } from 'surrealdb';
import * as SurrealNodeModule from '@surrealdb/node';
import { pipeline } from '@xenova/transformers';
import { mkdirSync, existsSync } from 'fs';
import { dirname, isAbsolute } from 'path';

import { SCHEMA_V1, SCHEMA_VERSION } from './schema.js';
import { StorageError } from './error.js';
import { Logger } from '../utils/logger.js';
import type {
  StorageConfig,
  SemanticConcept,
  DeveloperPattern,
  FileIntelligence,
  AIInsight,
  FeatureMap,
  EntryPoint,
  KeyDirectory,
  WorkSession,
  ProjectDecision,
  ProjectMetadata,
  CodeMetadata,
  SemanticSearchResult,
} from './types.js';
import { DEFAULT_STORAGE_CONFIG } from './types.js';

// ── Helper: coerce SurrealDB datetime strings/objects to JS Date ───────

function toDate(value: unknown): Date {
  if (value instanceof Date) return value;
  if (typeof value === 'string') return new Date(value);
  if (value && typeof value === 'object' && 'toISOString' in value) {
    return new Date((value as { toISOString(): string }).toISOString());
  }
  return new Date();
}

function toOptionalDate(value: unknown): Date | undefined {
  if (value == null) return undefined;
  return toDate(value);
}

// ── Helper: extract SurrealDB record id as string ──────────────────────

function recordId(raw: unknown): string {
  if (raw && typeof raw === 'object') {
    // SurrealDB RecordId: .id getter returns the id part (array for composite keys)
    const idPart = (raw as any).id;
    if (Array.isArray(idPart)) return String(idPart[0]);
    if (idPart !== undefined) return String(idPart);
  }
  if (typeof raw === 'string') {
    // May arrive as "table:id" or "table:['id']" — strip table prefix
    const colonIdx = raw.indexOf(':');
    if (colonIdx >= 0) {
      let id = raw.slice(colonIdx + 1).trim();
      id = id.replace(/^\[?\s*s?"?'?/, '').replace(/"?'?\s*\]?$/, '');
      return id;
    }
    return raw;
  }
  return String(raw);
}

export class SurrealStorage {
  private db: Surreal;
  private config: StorageConfig;
  private localEmbeddingPipeline: any;
  private embeddingCache = new Map<string, number[]>();
  private readonly EMBEDDING_CACHE_SIZE = 1000;
  private hasLoggedEmbeddingStart = false;

  // ── Factory + lifecycle ────────────────────────────────────────────

  private constructor(db: Surreal, config: StorageConfig) {
    this.db = db;
    this.config = config;
  }

  /**
   * Async factory — the only way to obtain a SurrealStorage instance.
   */
  static async create(
    dbPath: string,
    configOverrides?: Partial<StorageConfig>,
  ): Promise<SurrealStorage> {
    const cfg: StorageConfig = { ...DEFAULT_STORAGE_CONFIG, ...configOverrides };

    // Ensure parent directory exists for file-based databases
    if (dbPath !== ':memory:') {
      const dir = dirname(dbPath);
      if (!existsSync(dir)) {
        Logger.info(`Creating database directory: ${dir}`);
        mkdirSync(dir, { recursive: true });
      }
    }

    const createEngines =
      (SurrealNodeModule as any).createNodeEngines ??
      (SurrealNodeModule as any).surrealdbNodeEngines;

    const db = new Surreal({ engines: createEngines() });

    try {
      await db.connect(`surrealkv://${dbPath}`);
    } catch (cause) {
      throw StorageError.connectionFailed(cause);
    }

    await db.use({ namespace: cfg.namespace, database: cfg.database });

    const storage = new SurrealStorage(db, cfg);

    // Apply schema idempotently
    try {
      await db.query(SCHEMA_V1);
    } catch (cause) {
      throw StorageError.schemaFailed(cause);
    }

    // Stamp schema version (upsert)
    await storage.stampSchemaVersion();

    // Start embedding pipeline in background (non-blocking)
    storage.initializeLocalEmbeddings();

    Logger.info('SurrealStorage initialized successfully');
    return storage;
  }

  async close(): Promise<void> {
    if (this.localEmbeddingPipeline) {
      try {
        if (typeof this.localEmbeddingPipeline.dispose === 'function') {
          await this.localEmbeddingPipeline.dispose();
        }
        this.localEmbeddingPipeline = null;
      } catch (error) {
        Logger.warn('Failed to dispose local embedding pipeline:', error);
      }
    }
    await this.db.close();
  }

  async healthCheck(): Promise<void> {
    try {
      await this.db.query('SELECT * FROM schema_version LIMIT 1');
    } catch (cause) {
      throw StorageError.queryFailed('healthCheck', cause);
    }
  }

  // ── Schema version ────────────────────────────────────────────────

  private async stampSchemaVersion(): Promise<void> {
    // Delete old version records and insert current
    await this.db.query('DELETE schema_version');
    await this.db.query(
      'CREATE schema_version SET version = $version, applied = time::now()',
      { version: SCHEMA_VERSION },
    );
  }

  async getSchemaVersion(): Promise<number> {
    const result = await this.db.query<[{ version: number }[]]>(
      'SELECT version, applied FROM schema_version ORDER BY applied DESC LIMIT 1',
    );
    const rows = result[0] ?? [];
    return rows.length > 0 ? rows[0].version : 0;
  }

  // ── Semantic Concepts ─────────────────────────────────────────────

  async insertSemanticConcept(
    concept: Omit<SemanticConcept, 'createdAt' | 'updatedAt'>,
  ): Promise<void> {
    try {
      // Upsert by deleting existing + create
      await this.db.query('DELETE semantic_concept WHERE meta::id(id) = $id', {
        id: concept.id,
      });
      await this.db.query(
        `CREATE semantic_concept:[$id] SET
          concept_name = $concept_name,
          concept_type = $concept_type,
          confidence_score = $confidence_score,
          relationships = $relationships,
          evolution_history = $evolution_history,
          file_path = $file_path,
          line_range = $line_range,
          created_at = time::now(),
          updated_at = time::now()`,
        {
          id: concept.id,
          concept_name: concept.conceptName,
          concept_type: concept.conceptType,
          confidence_score: concept.confidenceScore,
          relationships: concept.relationships ?? {},
          evolution_history: concept.evolutionHistory ?? {},
          file_path: concept.filePath,
          line_range: concept.lineRange ?? { start: 0, end: 0 },
        },
      );
    } catch (cause) {
      throw StorageError.queryFailed('insertSemanticConcept', cause);
    }
  }

  async getSemanticConcepts(filePath?: string): Promise<SemanticConcept[]> {
    try {
      let rows: any[];
      if (filePath) {
        const result = await this.db.query<[any[]]>(
          'SELECT * FROM semantic_concept WHERE file_path = $file_path',
          { file_path: filePath },
        );
        rows = result[0] ?? [];
      } else {
        const result = await this.db.query<[any[]]>(
          'SELECT * FROM semantic_concept',
        );
        rows = result[0] ?? [];
      }
      return rows.map(this.mapSemanticConcept);
    } catch (cause) {
      throw StorageError.queryFailed('getSemanticConcepts', cause);
    }
  }

  private mapSemanticConcept(row: any): SemanticConcept {
    return {
      id: recordId(row.id),
      conceptName: row.concept_name,
      conceptType: row.concept_type,
      confidenceScore: row.confidence_score ?? 0,
      relationships: row.relationships ?? {},
      evolutionHistory: row.evolution_history ?? {},
      filePath: row.file_path,
      lineRange: row.line_range ?? { start: 0, end: 0 },
      createdAt: toDate(row.created_at),
      updatedAt: toDate(row.updated_at),
    };
  }

  // ── Developer Patterns ────────────────────────────────────────────

  async insertDeveloperPattern(
    pattern: Omit<DeveloperPattern, 'createdAt' | 'lastSeen'>,
  ): Promise<void> {
    try {
      await this.db.query(
        'DELETE developer_pattern WHERE meta::id(id) = $id',
        { id: pattern.patternId },
      );
      await this.db.query(
        `CREATE developer_pattern:[$id] SET
          pattern_type = $pattern_type,
          pattern_content = $pattern_content,
          frequency = $frequency,
          contexts = $contexts,
          examples = $examples,
          confidence = $confidence,
          created_at = time::now(),
          last_seen = time::now()`,
        {
          id: pattern.patternId,
          pattern_type: pattern.patternType,
          pattern_content: pattern.patternContent,
          frequency: pattern.frequency,
          contexts: pattern.contexts ?? [],
          examples: pattern.examples ?? [],
          confidence: pattern.confidence,
        },
      );
    } catch (cause) {
      throw StorageError.queryFailed('insertDeveloperPattern', cause);
    }
  }

  async getDeveloperPatterns(
    patternType?: string,
    limit?: number,
  ): Promise<DeveloperPattern[]> {
    try {
      const effectiveLimit = limit !== undefined && limit > 0 ? limit : 50;
      let result: any;

      if (patternType) {
        result = await this.db.query<[any[]]>(
          `SELECT * FROM developer_pattern
           WHERE pattern_type = $pattern_type
           ORDER BY frequency DESC, confidence DESC
           LIMIT $limit`,
          { pattern_type: patternType, limit: effectiveLimit },
        );
      } else {
        result = await this.db.query<[any[]]>(
          `SELECT * FROM developer_pattern
           ORDER BY frequency DESC, confidence DESC
           LIMIT $limit`,
          { limit: effectiveLimit },
        );
      }

      const rows = result[0] ?? [];
      return rows.map(this.mapDeveloperPattern);
    } catch (cause) {
      throw StorageError.queryFailed('getDeveloperPatterns', cause);
    }
  }

  private mapDeveloperPattern(row: any): DeveloperPattern {
    return {
      patternId: recordId(row.id),
      patternType: row.pattern_type,
      patternContent: row.pattern_content ?? {},
      frequency: row.frequency ?? 1,
      contexts: row.contexts ?? [],
      examples: row.examples ?? [],
      confidence: row.confidence ?? 0,
      createdAt: toDate(row.created_at),
      lastSeen: toDate(row.last_seen),
    };
  }

  // ── File Intelligence ─────────────────────────────────────────────

  async insertFileIntelligence(
    fileIntel: Omit<FileIntelligence, 'createdAt'>,
  ): Promise<void> {
    try {
      // Upsert by file_path (unique)
      await this.db.query(
        'DELETE file_intelligence WHERE file_path = $file_path',
        { file_path: fileIntel.filePath },
      );
      await this.db.query(
        `CREATE file_intelligence SET
          file_path = $file_path,
          file_hash = $file_hash,
          semantic_concepts = $semantic_concepts,
          patterns_used = $patterns_used,
          complexity_metrics = $complexity_metrics,
          dependencies = $dependencies,
          last_analyzed = time::now(),
          created_at = time::now()`,
        {
          file_path: fileIntel.filePath,
          file_hash: fileIntel.fileHash,
          semantic_concepts: fileIntel.semanticConcepts ?? [],
          patterns_used: fileIntel.patternsUsed ?? [],
          complexity_metrics: fileIntel.complexityMetrics ?? {},
          dependencies: fileIntel.dependencies ?? [],
        },
      );
    } catch (cause) {
      throw StorageError.queryFailed('insertFileIntelligence', cause);
    }
  }

  async getFileIntelligence(filePath: string): Promise<FileIntelligence | null> {
    try {
      const result = await this.db.query<[any[]]>(
        'SELECT * FROM file_intelligence WHERE file_path = $file_path LIMIT 1',
        { file_path: filePath },
      );
      const rows = result[0] ?? [];
      if (rows.length === 0) return null;
      const row = rows[0];
      return {
        filePath: row.file_path,
        fileHash: row.file_hash,
        semanticConcepts: row.semantic_concepts ?? [],
        patternsUsed: row.patterns_used ?? [],
        complexityMetrics: row.complexity_metrics ?? {},
        dependencies: row.dependencies ?? [],
        lastAnalyzed: toDate(row.last_analyzed),
        createdAt: toDate(row.created_at),
      };
    } catch (cause) {
      throw StorageError.queryFailed('getFileIntelligence', cause);
    }
  }

  // ── AI Insights ───────────────────────────────────────────────────

  async insertAIInsight(
    insight: Omit<AIInsight, 'createdAt'>,
  ): Promise<void> {
    try {
      await this.db.query(
        `CREATE ai_insight:[$id] SET
          insight_type = $insight_type,
          insight_content = $insight_content,
          confidence_score = $confidence_score,
          source_agent = $source_agent,
          validation_status = $validation_status,
          impact_prediction = $impact_prediction,
          created_at = time::now()`,
        {
          id: insight.insightId,
          insight_type: insight.insightType,
          insight_content: insight.insightContent,
          confidence_score: insight.confidenceScore,
          source_agent: insight.sourceAgent,
          validation_status: insight.validationStatus,
          impact_prediction: insight.impactPrediction ?? {},
        },
      );
    } catch (cause) {
      throw StorageError.queryFailed('insertAIInsight', cause);
    }
  }

  async getAIInsights(insightType?: string): Promise<AIInsight[]> {
    try {
      let result: any;
      if (insightType) {
        result = await this.db.query<[any[]]>(
          `SELECT * FROM ai_insight
           WHERE insight_type = $insight_type
           ORDER BY confidence_score DESC, created_at DESC`,
          { insight_type: insightType },
        );
      } else {
        result = await this.db.query<[any[]]>(
          'SELECT * FROM ai_insight ORDER BY confidence_score DESC, created_at DESC',
        );
      }
      const rows = result[0] ?? [];
      return rows.map((row: any): AIInsight => ({
        insightId: recordId(row.id),
        insightType: row.insight_type,
        insightContent: row.insight_content ?? {},
        confidenceScore: row.confidence_score ?? 0,
        sourceAgent: row.source_agent,
        validationStatus: row.validation_status as AIInsight['validationStatus'],
        impactPrediction: row.impact_prediction ?? {},
        createdAt: toDate(row.created_at),
      }));
    } catch (cause) {
      throw StorageError.queryFailed('getAIInsights', cause);
    }
  }

  // ── Feature Maps ──────────────────────────────────────────────────

  async insertFeatureMap(
    feature: Omit<FeatureMap, 'createdAt' | 'updatedAt'>,
  ): Promise<void> {
    try {
      await this.db.query('DELETE feature_map WHERE meta::id(id) = $id', {
        id: feature.id,
      });
      await this.db.query(
        `CREATE feature_map:[$id] SET
          project_path = $project_path,
          feature_name = $feature_name,
          primary_files = $primary_files,
          related_files = $related_files,
          dependencies = $dependencies,
          status = $status,
          created_at = time::now(),
          updated_at = time::now()`,
        {
          id: feature.id,
          project_path: feature.projectPath,
          feature_name: feature.featureName,
          primary_files: feature.primaryFiles ?? [],
          related_files: feature.relatedFiles ?? [],
          dependencies: feature.dependencies ?? [],
          status: feature.status ?? 'active',
        },
      );
    } catch (cause) {
      throw StorageError.queryFailed('insertFeatureMap', cause);
    }
  }

  async getFeatureMaps(projectPath: string): Promise<FeatureMap[]> {
    try {
      // Normalize path — try both absolute and relative (.) paths
      const paths = [projectPath];
      if (isAbsolute(projectPath)) paths.push('.');

      for (const p of paths) {
        const result = await this.db.query<[any[]]>(
          `SELECT * FROM feature_map
           WHERE project_path = $project_path AND status = 'active'
           ORDER BY feature_name`,
          { project_path: p },
        );
        const rows = result[0] ?? [];
        if (rows.length > 0) return rows.map(this.mapFeatureMap);
      }
      return [];
    } catch (cause) {
      throw StorageError.queryFailed('getFeatureMaps', cause);
    }
  }

  async searchFeatureMaps(
    projectPath: string,
    query: string,
  ): Promise<FeatureMap[]> {
    try {
      const result = await this.db.query<[any[]]>(
        `SELECT * FROM feature_map
         WHERE project_path = $project_path AND status = 'active'
           AND string::lowercase(feature_name) CONTAINS string::lowercase($query)
         ORDER BY feature_name`,
        { project_path: projectPath, query },
      );
      const rows = result[0] ?? [];
      return rows.map(this.mapFeatureMap);
    } catch (cause) {
      throw StorageError.queryFailed('searchFeatureMaps', cause);
    }
  }

  async getFeatureByName(
    projectPath: string,
    featureName: string,
  ): Promise<FeatureMap | null> {
    try {
      const result = await this.db.query<[any[]]>(
        `SELECT * FROM feature_map
         WHERE project_path = $project_path AND feature_name = $feature_name AND status = 'active'
         LIMIT 1`,
        { project_path: projectPath, feature_name: featureName },
      );
      const rows = result[0] ?? [];
      return rows.length > 0 ? this.mapFeatureMap(rows[0]) : null;
    } catch (cause) {
      throw StorageError.queryFailed('getFeatureByName', cause);
    }
  }

  private mapFeatureMap(row: any): FeatureMap {
    return {
      id: recordId(row.id),
      projectPath: row.project_path,
      featureName: row.feature_name,
      primaryFiles: row.primary_files ?? [],
      relatedFiles: row.related_files ?? [],
      dependencies: row.dependencies ?? [],
      status: row.status ?? 'active',
      createdAt: toDate(row.created_at),
      updatedAt: toDate(row.updated_at),
    };
  }

  // ── Entry Points ──────────────────────────────────────────────────

  async insertEntryPoint(
    entryPoint: Omit<EntryPoint, 'createdAt'>,
  ): Promise<void> {
    try {
      await this.db.query('DELETE entry_point WHERE meta::id(id) = $id', {
        id: entryPoint.id,
      });
      const sets = [
        'project_path = $project_path',
        'entry_type = $entry_type',
        'file_path = $file_path',
        'created_at = time::now()',
      ];
      const params: Record<string, unknown> = {
        id: entryPoint.id,
        project_path: entryPoint.projectPath,
        entry_type: entryPoint.entryType,
        file_path: entryPoint.filePath,
      };
      if (entryPoint.description !== undefined) {
        sets.push('description = $description');
        params.description = entryPoint.description;
      }
      if (entryPoint.framework !== undefined) {
        sets.push('framework = $framework');
        params.framework = entryPoint.framework;
      }
      await this.db.query(
        `CREATE entry_point:[$id] SET ${sets.join(', ')}`,
        params,
      );
    } catch (cause) {
      throw StorageError.queryFailed('insertEntryPoint', cause);
    }
  }

  async getEntryPoints(projectPath: string): Promise<EntryPoint[]> {
    try {
      const paths = [projectPath];
      if (isAbsolute(projectPath)) paths.push('.');

      for (const p of paths) {
        const result = await this.db.query<[any[]]>(
          `SELECT * FROM entry_point
           WHERE project_path = $project_path
           ORDER BY entry_type, file_path`,
          { project_path: p },
        );
        const rows = result[0] ?? [];
        if (rows.length > 0) {
          return rows.map((row: any): EntryPoint => ({
            id: recordId(row.id),
            projectPath: row.project_path,
            entryType: row.entry_type,
            filePath: row.file_path,
            description: row.description ?? undefined,
            framework: row.framework ?? undefined,
            createdAt: toDate(row.created_at),
          }));
        }
      }
      return [];
    } catch (cause) {
      throw StorageError.queryFailed('getEntryPoints', cause);
    }
  }

  // ── Key Directories ───────────────────────────────────────────────

  async insertKeyDirectory(
    directory: Omit<KeyDirectory, 'createdAt'>,
  ): Promise<void> {
    try {
      await this.db.query('DELETE key_directory WHERE meta::id(id) = $id', {
        id: directory.id,
      });
      const sets = [
        'project_path = $project_path',
        'directory_path = $directory_path',
        'directory_type = $directory_type',
        'file_count = $file_count',
        'created_at = time::now()',
      ];
      const params: Record<string, unknown> = {
        id: directory.id,
        project_path: directory.projectPath,
        directory_path: directory.directoryPath,
        directory_type: directory.directoryType,
        file_count: directory.fileCount,
      };
      if (directory.description !== undefined) {
        sets.push('description = $description');
        params.description = directory.description;
      }
      await this.db.query(
        `CREATE key_directory:[$id] SET ${sets.join(', ')}`,
        params,
      );
    } catch (cause) {
      throw StorageError.queryFailed('insertKeyDirectory', cause);
    }
  }

  async getKeyDirectories(projectPath: string): Promise<KeyDirectory[]> {
    try {
      const paths = [projectPath];
      if (isAbsolute(projectPath)) paths.push('.');

      for (const p of paths) {
        const result = await this.db.query<[any[]]>(
          `SELECT * FROM key_directory
           WHERE project_path = $project_path
           ORDER BY directory_type, directory_path`,
          { project_path: p },
        );
        const rows = result[0] ?? [];
        if (rows.length > 0) {
          return rows.map((row: any): KeyDirectory => ({
            id: recordId(row.id),
            projectPath: row.project_path,
            directoryPath: row.directory_path,
            directoryType: row.directory_type,
            fileCount: row.file_count ?? 0,
            description: row.description ?? undefined,
            createdAt: toDate(row.created_at),
          }));
        }
      }
      return [];
    } catch (cause) {
      throw StorageError.queryFailed('getKeyDirectories', cause);
    }
  }

  // ── Project Metadata ──────────────────────────────────────────────

  async insertProjectMetadata(metadata: {
    projectId: string;
    projectPath: string;
    projectName?: string;
    languagePrimary?: string;
    languagesDetected?: string[];
    frameworkDetected?: string[];
    intelligenceVersion?: string;
    lastFullScan?: Date;
  }): Promise<void> {
    try {
      // Upsert by project_path (unique index)
      await this.db.query(
        'DELETE project_metadata WHERE project_path = $project_path',
        { project_path: metadata.projectPath },
      );

      const sets = [
        'project_path = $project_path',
        'languages_detected = $languages_detected',
        'framework_detected = $framework_detected',
        'created_at = time::now()',
        'updated_at = time::now()',
      ];
      const params: Record<string, unknown> = {
        id: metadata.projectId,
        project_path: metadata.projectPath,
        languages_detected: metadata.languagesDetected ?? [],
        framework_detected: metadata.frameworkDetected ?? [],
      };

      if (metadata.projectName !== undefined) {
        sets.push('project_name = $project_name');
        params.project_name = metadata.projectName;
      }
      if (metadata.languagePrimary !== undefined) {
        sets.push('language_primary = $language_primary');
        params.language_primary = metadata.languagePrimary;
      }
      if (metadata.intelligenceVersion !== undefined) {
        sets.push('intelligence_version = $intelligence_version');
        params.intelligence_version = metadata.intelligenceVersion;
      }
      if (metadata.lastFullScan !== undefined) {
        sets.push('last_full_scan = <datetime>$last_full_scan');
        params.last_full_scan = metadata.lastFullScan.toISOString();
      }

      await this.db.query(
        `CREATE project_metadata:[$id] SET ${sets.join(', ')}`,
        params,
      );
    } catch (cause) {
      throw StorageError.queryFailed('insertProjectMetadata', cause);
    }
  }

  async getProjectMetadata(projectPath: string): Promise<ProjectMetadata | null> {
    try {
      const result = await this.db.query<[any[]]>(
        'SELECT * FROM project_metadata WHERE project_path = $project_path LIMIT 1',
        { project_path: projectPath },
      );
      const rows = result[0] ?? [];
      if (rows.length === 0) return null;
      const row = rows[0];
      return {
        projectId: recordId(row.id),
        projectPath: row.project_path,
        projectName: row.project_name ?? undefined,
        languagePrimary: row.language_primary ?? undefined,
        languagesDetected: row.languages_detected ?? [],
        frameworkDetected: row.framework_detected ?? [],
        intelligenceVersion: row.intelligence_version ?? undefined,
        lastFullScan: toOptionalDate(row.last_full_scan),
        createdAt: toDate(row.created_at),
        updatedAt: toDate(row.updated_at),
      };
    } catch (cause) {
      throw StorageError.queryFailed('getProjectMetadata', cause);
    }
  }

  // ── Work Sessions ─────────────────────────────────────────────────

  async createWorkSession(
    session: Omit<WorkSession, 'sessionStart' | 'lastUpdated'>,
  ): Promise<void> {
    try {
      // Build SET clause dynamically, omitting optional fields that are undefined
      const sets = [
        'project_path = $project_path',
        'session_start = time::now()',
        'current_files = $current_files',
        'completed_tasks = $completed_tasks',
        'pending_tasks = $pending_tasks',
        'blockers = $blockers',
        'last_updated = time::now()',
      ];
      const params: Record<string, unknown> = {
        id: session.id,
        project_path: session.projectPath,
        current_files: session.currentFiles ?? [],
        completed_tasks: session.completedTasks ?? [],
        pending_tasks: session.pendingTasks ?? [],
        blockers: session.blockers ?? [],
      };

      if (session.sessionEnd !== undefined) {
        sets.push('session_end = $session_end');
        params.session_end = session.sessionEnd.toISOString();
      }
      if (session.lastFeature !== undefined) {
        sets.push('last_feature = $last_feature');
        params.last_feature = session.lastFeature;
      }
      if (session.sessionNotes !== undefined) {
        sets.push('session_notes = $session_notes');
        params.session_notes = session.sessionNotes;
      }

      await this.db.query(
        `CREATE work_session:[$id] SET ${sets.join(', ')}`,
        params,
      );
    } catch (cause) {
      throw StorageError.queryFailed('createWorkSession', cause);
    }
  }

  async updateWorkSession(
    sessionId: string,
    updates: Partial<Omit<WorkSession, 'id' | 'projectPath' | 'sessionStart' | 'lastUpdated'>>,
  ): Promise<void> {
    try {
      const sets: string[] = [];
      const params: Record<string, unknown> = { id: sessionId };

      if (updates.sessionEnd !== undefined) {
        sets.push('session_end = <datetime>$session_end');
        params.session_end = updates.sessionEnd.toISOString();
      }
      if (updates.lastFeature !== undefined) {
        sets.push('last_feature = $last_feature');
        params.last_feature = updates.lastFeature;
      }
      if (updates.currentFiles !== undefined) {
        sets.push('current_files = $current_files');
        params.current_files = updates.currentFiles;
      }
      if (updates.completedTasks !== undefined) {
        sets.push('completed_tasks = $completed_tasks');
        params.completed_tasks = updates.completedTasks;
      }
      if (updates.pendingTasks !== undefined) {
        sets.push('pending_tasks = $pending_tasks');
        params.pending_tasks = updates.pendingTasks;
      }
      if (updates.blockers !== undefined) {
        sets.push('blockers = $blockers');
        params.blockers = updates.blockers;
      }
      if (updates.sessionNotes !== undefined) {
        sets.push('session_notes = $session_notes');
        params.session_notes = updates.sessionNotes;
      }

      if (sets.length === 0) return;
      sets.push('last_updated = time::now()');

      await this.db.query(
        `UPDATE work_session SET ${sets.join(', ')} WHERE meta::id(id) = $id`,
        params,
      );
    } catch (cause) {
      throw StorageError.queryFailed('updateWorkSession', cause);
    }
  }

  async getCurrentWorkSession(projectPath: string): Promise<WorkSession | null> {
    try {
      const result = await this.db.query<[any[]]>(
        `SELECT * FROM work_session
         WHERE project_path = $project_path AND session_end IS NONE
         ORDER BY session_start DESC
         LIMIT 1`,
        { project_path: projectPath },
      );
      const rows = result[0] ?? [];
      return rows.length > 0 ? this.mapWorkSession(rows[0]) : null;
    } catch (cause) {
      throw StorageError.queryFailed('getCurrentWorkSession', cause);
    }
  }

  async getWorkSessions(
    projectPath: string,
    limit = 10,
  ): Promise<WorkSession[]> {
    try {
      const result = await this.db.query<[any[]]>(
        `SELECT * FROM work_session
         WHERE project_path = $project_path
         ORDER BY session_start DESC
         LIMIT $limit`,
        { project_path: projectPath, limit },
      );
      const rows = result[0] ?? [];
      return rows.map(this.mapWorkSession);
    } catch (cause) {
      throw StorageError.queryFailed('getWorkSessions', cause);
    }
  }

  private mapWorkSession(row: any): WorkSession {
    return {
      id: recordId(row.id),
      projectPath: row.project_path,
      sessionStart: toDate(row.session_start),
      sessionEnd: toOptionalDate(row.session_end),
      lastFeature: row.last_feature ?? undefined,
      currentFiles: row.current_files ?? [],
      completedTasks: row.completed_tasks ?? [],
      pendingTasks: row.pending_tasks ?? [],
      blockers: row.blockers ?? [],
      sessionNotes: row.session_notes ?? undefined,
      lastUpdated: toDate(row.last_updated),
    };
  }

  // ── Project Decisions ─────────────────────────────────────────────

  async upsertProjectDecision(
    decision: Omit<ProjectDecision, 'madeAt'>,
  ): Promise<void> {
    try {
      // Delete existing decision with same project_path + decision_key
      await this.db.query(
        `DELETE project_decision
         WHERE project_path = $project_path AND decision_key = $decision_key`,
        { project_path: decision.projectPath, decision_key: decision.decisionKey },
      );
      const sets = [
        'project_path = $project_path',
        'decision_key = $decision_key',
        'decision_value = $decision_value',
        'made_at = time::now()',
      ];
      const params: Record<string, unknown> = {
        id: decision.id,
        project_path: decision.projectPath,
        decision_key: decision.decisionKey,
        decision_value: decision.decisionValue,
      };
      if (decision.reasoning !== undefined) {
        sets.push('reasoning = $reasoning');
        params.reasoning = decision.reasoning;
      }
      await this.db.query(
        `CREATE project_decision:[$id] SET ${sets.join(', ')}`,
        params,
      );
    } catch (cause) {
      throw StorageError.queryFailed('upsertProjectDecision', cause);
    }
  }

  async getProjectDecisions(
    projectPath: string,
    limit = 20,
  ): Promise<ProjectDecision[]> {
    try {
      const result = await this.db.query<[any[]]>(
        `SELECT * FROM project_decision
         WHERE project_path = $project_path
         ORDER BY made_at DESC
         LIMIT $limit`,
        { project_path: projectPath, limit },
      );
      const rows = result[0] ?? [];
      return rows.map(this.mapProjectDecision);
    } catch (cause) {
      throw StorageError.queryFailed('getProjectDecisions', cause);
    }
  }

  async getProjectDecision(
    projectPath: string,
    decisionKey: string,
  ): Promise<ProjectDecision | null> {
    try {
      const result = await this.db.query<[any[]]>(
        `SELECT * FROM project_decision
         WHERE project_path = $project_path AND decision_key = $decision_key
         LIMIT 1`,
        { project_path: projectPath, decision_key: decisionKey },
      );
      const rows = result[0] ?? [];
      return rows.length > 0 ? this.mapProjectDecision(rows[0]) : null;
    } catch (cause) {
      throw StorageError.queryFailed('getProjectDecision', cause);
    }
  }

  private mapProjectDecision(row: any): ProjectDecision {
    return {
      id: recordId(row.id),
      projectPath: row.project_path,
      decisionKey: row.decision_key,
      decisionValue: row.decision_value,
      reasoning: row.reasoning ?? undefined,
      madeAt: toDate(row.made_at),
    };
  }

  // ── Code Embeddings (from SemanticVectorDB) ───────────────────────

  async storeCodeEmbedding(
    code: string,
    metadata: CodeMetadata,
  ): Promise<void> {
    try {
      const embedding = await this.generateEmbedding(code);
      await this.db.query(
        `CREATE code_document SET
          code = $code,
          embedding = $embedding,
          metadata = $metadata,
          created_at = time::now(),
          updated_at = time::now()`,
        { code, embedding, metadata },
      );
    } catch (cause) {
      throw StorageError.queryFailed('storeCodeEmbedding', cause);
    }
  }

  async storeMultipleEmbeddings(
    codeChunks: string[],
    metadataList: CodeMetadata[],
  ): Promise<void> {
    if (codeChunks.length !== metadataList.length) {
      throw new Error('Code chunks and metadata arrays must have the same length');
    }

    for (let i = 0; i < codeChunks.length; i++) {
      await this.storeCodeEmbedding(codeChunks[i], metadataList[i]);
    }
  }

  async findSimilarCode(
    query: string,
    limit = 5,
    filters?: Record<string, unknown>,
  ): Promise<SemanticSearchResult[]> {
    try {
      if (!query || query.trim() === '') {
        // No query — return all documents matching filters
        let searchQuery = 'SELECT * FROM code_document';
        const params: Record<string, unknown> = { limit };

        if (filters) {
          const conditions = Object.entries(filters)
            .map(([key]) => `metadata.${key} = $${key}`)
            .join(' AND ');
          searchQuery += ` WHERE ${conditions}`;
          Object.assign(params, filters);
        }

        searchQuery += ' LIMIT $limit';
        const result = await this.db.query<[any[]]>(searchQuery, params);
        const rows = result[0] ?? [];

        return rows.map((doc: any): SemanticSearchResult => ({
          id: recordId(doc.id),
          code: doc.code,
          metadata: doc.metadata,
          similarity: 0.5,
        }));
      }

      // Use SurrealDB full-text search for BM25 scoring
      let searchQuery = `
        SELECT *, search::score(1) AS similarity
        FROM code_document
        WHERE code @@ $query`;

      const params: Record<string, unknown> = { query, limit };

      if (filters) {
        const conditions = Object.entries(filters)
          .map(([key]) => `metadata.${key} = $${key}`)
          .join(' AND ');
        searchQuery += ` AND ${conditions}`;
        Object.assign(params, filters);
      }

      searchQuery += ' ORDER BY similarity DESC LIMIT $limit';

      const result = await this.db.query<[any[]]>(searchQuery, params);
      const rows = result[0] ?? [];

      return rows.map((doc: any): SemanticSearchResult => ({
        id: recordId(doc.id),
        code: doc.code,
        metadata: doc.metadata,
        similarity: doc.similarity ?? 0,
      }));
    } catch (cause) {
      throw StorageError.queryFailed('findSimilarCode', cause);
    }
  }

  async findSimilarCodeByFile(
    filePath: string,
    limit = 5,
  ): Promise<SemanticSearchResult[]> {
    return this.findSimilarCode('', limit, { filePath });
  }

  async findSimilarCodeByLanguage(
    query: string,
    language: string,
    limit = 5,
  ): Promise<SemanticSearchResult[]> {
    return this.findSimilarCode(query, limit, { language });
  }

  async updateCodeEmbedding(
    id: string,
    code: string,
    metadata: CodeMetadata,
  ): Promise<void> {
    try {
      const embedding = await this.generateEmbedding(code);
      await this.db.query(
        `UPDATE $id SET
          code = $code,
          embedding = $embedding,
          metadata = $metadata,
          updated_at = time::now()`,
        { id, code, embedding, metadata },
      );
    } catch (cause) {
      throw StorageError.queryFailed('updateCodeEmbedding', cause);
    }
  }

  async deleteCodeEmbedding(id: string): Promise<void> {
    try {
      await this.db.query('DELETE $id', { id });
    } catch (cause) {
      throw StorageError.queryFailed('deleteCodeEmbedding', cause);
    }
  }

  async deleteCodeEmbeddingsByFile(filePath: string): Promise<void> {
    try {
      await this.db.query(
        'DELETE code_document WHERE metadata.filePath = $filePath',
        { filePath },
      );
    } catch (cause) {
      throw StorageError.queryFailed('deleteCodeEmbeddingsByFile', cause);
    }
  }

  async getCollectionStats(): Promise<{ count: number; metadata: any }> {
    try {
      const result = await this.db.query<[any[]]>(
        'SELECT count() AS total FROM code_document GROUP ALL',
      );
      const rows = result[0] ?? [];
      const count = rows.length > 0 ? (rows[0] as any).total ?? 0 : 0;

      return {
        count,
        metadata: {
          description: 'Anime Mori semantic code embeddings',
          engine: 'SurrealDB',
        },
      };
    } catch (cause) {
      throw StorageError.queryFailed('getCollectionStats', cause);
    }
  }

  // ── Embedding pipeline ────────────────────────────────────────────

  private async initializeLocalEmbeddings(): Promise<void> {
    try {
      Logger.info('Initializing local embedding pipeline...');
      this.localEmbeddingPipeline = await pipeline(
        'feature-extraction',
        'Xenova/all-MiniLM-L6-v2',
      );
      Logger.info('Local embedding pipeline ready');
    } catch (error) {
      Logger.warn(
        'Failed to initialize local embeddings:',
        error instanceof Error ? error.message : String(error),
      );
      Logger.info('Will use fallback local embedding method');
    }
  }

  private async generateEmbedding(text: string): Promise<number[]> {
    const cacheKey = this.createCacheKey(text);
    const cached = this.embeddingCache.get(cacheKey);
    if (cached) return cached;

    if (!this.hasLoggedEmbeddingStart) {
      Logger.info('Generating embeddings...');
      this.hasLoggedEmbeddingStart = true;
    }

    const embedding = await this.getLocalEmbedding(text);
    this.cacheEmbedding(cacheKey, embedding);
    return embedding;
  }

  private async getLocalEmbedding(code: string): Promise<number[]> {
    if (this.localEmbeddingPipeline) {
      try {
        const cleanCode = this.preprocessCodeForEmbedding(code);
        const result = await this.localEmbeddingPipeline(cleanCode, {
          pooling: 'mean',
          normalize: true,
        });
        return Array.from(result.data) as number[];
      } catch (error) {
        Logger.warn(
          'Local embedding pipeline failed:',
          error instanceof Error ? error.message : String(error),
        );
      }
    }
    return this.generateAdvancedLocalEmbedding(code);
  }

  // ── Advanced fallback embedding ───────────────────────────────────

  private generateAdvancedLocalEmbedding(code: string): number[] {
    const dim = this.config.vectorDimensions;
    const embedding = new Array(dim).fill(0);

    const structuralSize = Math.floor(dim * 0.25);
    const semanticSize = Math.floor(dim * 0.35);
    const astSize = Math.floor(dim * 0.25);
    const contextSize = dim - structuralSize - semanticSize - astSize;

    const structural = this.extractStructuralFeatures(code);
    for (let i = 0; i < Math.min(structuralSize, structural.length); i++) {
      embedding[i] = structural[i];
    }

    const semantic = this.extractSemanticFeatures(code);
    for (let i = 0; i < Math.min(semanticSize, semantic.length); i++) {
      embedding[structuralSize + i] = semantic[i];
    }

    const ast = this.extractASTFeatures(code);
    const astStart = structuralSize + semanticSize;
    for (let i = 0; i < Math.min(astSize, ast.length); i++) {
      embedding[astStart + i] = ast[i];
    }

    const context = this.extractContextFeatures(code);
    const contextStart = astStart + astSize;
    for (let i = 0; i < Math.min(contextSize, context.length); i++) {
      embedding[contextStart + i] = context[i];
    }

    return this.normalizeVector(embedding);
  }

  private extractStructuralFeatures(code: string): number[] {
    const features: number[] = [];

    const functions = (code.match(/function\s+\w+|const\s+\w+\s*=\s*(?:\([^)]*\)\s*=>|async\s*\([^)]*\)\s*=>)/g) || []).length;
    features.push(Math.min(functions / 10, 1));

    const classes = (code.match(/class\s+\w+/g) || []).length;
    features.push(Math.min(classes / 5, 1));

    const imports = (code.match(/import\s+.*from|export\s+/g) || []).length;
    features.push(Math.min(imports / 10, 1));

    const asyncCount = (code.match(/async\s+|await\s+|Promise/g) || []).length;
    features.push(Math.min(asyncCount / 8, 1));

    const control = (code.match(/if\s*\(|for\s*\(|while\s*\(|switch\s*\(/g) || []).length;
    features.push(Math.min(control / 15, 1));

    const patterns = [
      /try\s*{|catch\s*\(/g,
      /\.\w+\s*\(/g,
      /{\s*\w+:/g,
      /\[\w*\]/g,
      /=>\s*{/g,
      /interface\s+\w+/g,
      /type\s+\w+/g,
      /enum\s+\w+/g,
    ];

    for (const pattern of patterns) {
      const count = (code.match(pattern) || []).length;
      features.push(Math.min(count / 5, 1));
    }

    while (features.length < 96) features.push(0);
    return features.slice(0, 96);
  }

  private extractSemanticFeatures(code: string): number[] {
    const features: number[] = [];
    const tokens = this.extractMeaningfulTokens(code);

    const categories = [
      { keywords: ['service', 'controller', 'model', 'view', 'component'], weight: 1.0 },
      { keywords: ['create', 'read', 'update', 'delete', 'get', 'set'], weight: 0.9 },
      { keywords: ['user', 'auth', 'login', 'token', 'session'], weight: 0.8 },
      { keywords: ['api', 'http', 'request', 'response', 'endpoint'], weight: 0.8 },
      { keywords: ['database', 'query', 'table', 'schema', 'migration'], weight: 0.7 },
      { keywords: ['test', 'spec', 'mock', 'assert', 'expect'], weight: 0.7 },
      { keywords: ['config', 'env', 'settings', 'options'], weight: 0.6 },
      { keywords: ['util', 'helper', 'common', 'shared', 'lib'], weight: 0.5 },
    ];

    for (const category of categories) {
      let score = 0;
      for (const keyword of category.keywords) {
        score += tokens.filter(t => t.toLowerCase().includes(keyword.toLowerCase())).length * category.weight;
      }
      features.push(Math.min(score / 10, 1));
    }

    const vocab = this.getProgrammingVocabulary();
    const tokenFreq = this.calculateTokenFrequency(tokens);

    for (const term of vocab.slice(0, 120)) {
      const freq = tokenFreq.get(term.toLowerCase()) || 0;
      features.push(Math.min((freq / tokens.length) * 10, 1));
    }

    while (features.length < 134) features.push(0);
    return features.slice(0, 134);
  }

  private extractASTFeatures(code: string): number[] {
    const features: number[] = [];

    const declarations: Record<string, RegExp> = {
      variables: /(?:let|const|var)\s+\w+/g,
      functions: /function\s+\w+/g,
      classes: /class\s+\w+/g,
      interfaces: /interface\s+\w+/g,
    };

    for (const pattern of Object.values(declarations)) {
      features.push(Math.min((code.match(pattern) || []).length / 8, 1));
    }

    const expressions: Record<string, RegExp> = {
      assignments: /=\s*[^=]/g,
      comparisons: /[!=]==?|[<>]=?/g,
      logical: /&&|\|\|/g,
      arithmetic: /[+\-*/%]/g,
    };

    for (const pattern of Object.values(expressions)) {
      features.push(Math.min((code.match(pattern) || []).length / 20, 1));
    }

    let maxDepth = 0;
    let currentDepth = 0;
    for (const char of code) {
      if (char === '{') currentDepth++;
      if (char === '}') currentDepth--;
      maxDepth = Math.max(maxDepth, currentDepth);
    }
    features.push(Math.min(maxDepth / 8, 1));

    while (features.length < 96) features.push(0);
    return features.slice(0, 96);
  }

  private extractContextFeatures(code: string): number[] {
    const features: number[] = [];

    const comments = (code.match(/\/\/.*|\/\*[\s\S]*?\*\//g) || []).join('').length;
    features.push(Math.min(comments / code.length, 1));

    const strings = (code.match(/"[^"]*"|'[^']*'|`[^`]*`/g) || []).join('').length;
    features.push(Math.min(strings / code.length, 0.5));

    const lines = code.split('\n').length;
    const avgLineLength = code.length / lines;
    features.push(Math.min(lines / 100, 1));
    features.push(Math.min(avgLineLength / 80, 1));

    const domains: Record<string, RegExp> = {
      web: /http|url|fetch|ajax|xhr|dom|html|css/gi,
      database: /sql|query|select|insert|update|delete|join/gi,
      testing: /test|spec|describe|it|expect|assert|mock/gi,
      async: /async|await|promise|callback|then|catch/gi,
      security: /auth|encrypt|decrypt|hash|token|jwt|bcrypt/gi,
    };

    for (const pattern of Object.values(domains)) {
      features.push(Math.min((code.match(pattern) || []).length / 5, 1));
    }

    while (features.length < 58) features.push(0);
    return features.slice(0, 58);
  }

  private extractMeaningfulTokens(code: string): string[] {
    const cleanCode = code
      .replace(/\/\/.*$/gm, '')
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .replace(/["'`][^"'`]*["'`]/g, 'STRING');

    const tokens = cleanCode.match(/\b[a-zA-Z][a-zA-Z0-9_]*\b/g) || [];
    const noise = new Set(['a', 'an', 'the', 'is', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']);
    return tokens.filter(t => t.length > 2).filter(t => !noise.has(t.toLowerCase()));
  }

  private getProgrammingVocabulary(): string[] {
    return [
      'function', 'class', 'method', 'variable', 'constant', 'parameter', 'argument',
      'return', 'async', 'await', 'promise', 'callback', 'event', 'handler',
      'component', 'service', 'controller', 'model', 'view', 'router',
      'request', 'response', 'api', 'endpoint', 'middleware', 'auth',
      'database', 'query', 'select', 'insert', 'update', 'delete',
      'test', 'spec', 'mock', 'assert', 'expect', 'describe',
      'config', 'env', 'settings', 'options', 'params',
      'error', 'exception', 'try', 'catch', 'throw', 'finally',
      'loop', 'iteration', 'condition', 'branch', 'switch', 'case',
      'array', 'object', 'string', 'number', 'boolean', 'null',
      'import', 'export', 'module', 'require', 'include',
      'interface', 'type', 'generic', 'template', 'abstract',
      'static', 'private', 'public', 'protected', 'readonly',
      'constructor', 'destructor', 'extends', 'implements', 'super',
    ];
  }

  private calculateTokenFrequency(tokens: string[]): Map<string, number> {
    const freq = new Map<string, number>();
    for (const token of tokens) {
      const lower = token.toLowerCase();
      freq.set(lower, (freq.get(lower) || 0) + 1);
    }
    return freq;
  }

  private preprocessCodeForEmbedding(code: string): string {
    return code
      .replace(/\s+/g, ' ')
      .replace(/\/\/.*$/gm, '')
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .trim()
      .substring(0, 8000);
  }

  private createCacheKey(code: string): string {
    let hash = 0;
    for (let i = 0; i < Math.min(code.length, 1000); i++) {
      const char = code.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return hash.toString();
  }

  private cacheEmbedding(key: string, embedding: number[]): void {
    if (this.embeddingCache.size >= this.EMBEDDING_CACHE_SIZE) {
      const firstKey = this.embeddingCache.keys().next().value;
      if (firstKey !== undefined) this.embeddingCache.delete(firstKey);
    }
    this.embeddingCache.set(key, embedding);
  }

  private normalizeVector(vector: number[]): number[] {
    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    if (magnitude === 0) return vector;
    return vector.map(val => val / magnitude);
  }
}
