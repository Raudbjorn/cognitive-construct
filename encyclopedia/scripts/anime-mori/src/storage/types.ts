/**
 * Shared TypeScript interfaces for the Anime Mori storage layer.
 * Shared interfaces for the SurrealDB storage layer.
 */

// ── Storage configuration ──────────────────────────────────────────────

export interface StorageConfig {
  namespace: string;
  database: string;
  vectorDimensions: number;
}

export const DEFAULT_STORAGE_CONFIG: StorageConfig = {
  namespace: 'anime_mori',
  database: 'main',
  vectorDimensions: 384,
};

// ── Semantic Concepts ──────────────────────────────────────────────────

export interface SemanticConcept {
  id: string;
  conceptName: string;
  conceptType: string;
  confidenceScore: number;
  relationships: Record<string, unknown>;
  evolutionHistory: Record<string, unknown>;
  filePath: string;
  lineRange: { start: number; end: number };
  createdAt: Date;
  updatedAt: Date;
}

// ── Developer Patterns ─────────────────────────────────────────────────

export interface DeveloperPattern {
  patternId: string;
  patternType: string;
  patternContent: Record<string, unknown>;
  frequency: number;
  contexts: string[];
  examples: Record<string, unknown>[];
  confidence: number;
  createdAt: Date;
  lastSeen: Date;
}

// ── File Intelligence ──────────────────────────────────────────────────

export interface FileIntelligence {
  filePath: string;
  fileHash: string;
  semanticConcepts: string[];
  patternsUsed: string[];
  complexityMetrics: Record<string, number>;
  dependencies: string[];
  lastAnalyzed: Date;
  createdAt: Date;
}

// ── AI Insights ────────────────────────────────────────────────────────

export interface AIInsight {
  insightId: string;
  insightType: string;
  insightContent: Record<string, unknown>;
  confidenceScore: number;
  sourceAgent: string;
  validationStatus: 'pending' | 'validated' | 'rejected';
  impactPrediction: Record<string, unknown>;
  createdAt: Date;
}

// ── Feature Maps ───────────────────────────────────────────────────────

export interface FeatureMap {
  id: string;
  projectPath: string;
  featureName: string;
  primaryFiles: string[];
  relatedFiles: string[];
  dependencies: string[];
  status: string;
  createdAt: Date;
  updatedAt: Date;
}

// ── Entry Points ───────────────────────────────────────────────────────

export interface EntryPoint {
  id: string;
  projectPath: string;
  entryType: string;
  filePath: string;
  description?: string;
  framework?: string;
  createdAt: Date;
}

// ── Key Directories ────────────────────────────────────────────────────

export interface KeyDirectory {
  id: string;
  projectPath: string;
  directoryPath: string;
  directoryType: string;
  fileCount: number;
  description?: string;
  createdAt: Date;
}

// ── Work Sessions ──────────────────────────────────────────────────────

export interface WorkSession {
  id: string;
  projectPath: string;
  sessionStart: Date;
  sessionEnd?: Date;
  lastFeature?: string;
  currentFiles: string[];
  completedTasks: string[];
  pendingTasks: string[];
  blockers: string[];
  sessionNotes?: string;
  lastUpdated: Date;
}

// ── Project Decisions ──────────────────────────────────────────────────

export interface ProjectDecision {
  id: string;
  projectPath: string;
  decisionKey: string;
  decisionValue: string;
  reasoning?: string;
  madeAt: Date;
}

// ── Project Metadata ───────────────────────────────────────────────────

export interface ProjectMetadata {
  projectId: string;
  projectPath: string;
  projectName?: string;
  languagePrimary?: string;
  languagesDetected: string[];
  frameworkDetected: string[];
  intelligenceVersion?: string;
  lastFullScan?: Date;
  createdAt: Date;
  updatedAt: Date;
}

// ── Code Embeddings (from SemanticVectorDB) ────────────────────────────

export interface CodeMetadata {
  id: string;
  filePath: string;
  functionName?: string;
  className?: string;
  language: string;
  complexity: number;
  lineCount: number;
  lastModified: Date;
}

export interface SemanticSearchResult {
  id: string;
  code: string;
  metadata: CodeMetadata;
  similarity: number;
}
