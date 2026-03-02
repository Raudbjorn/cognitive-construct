import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { SurrealStorage } from '../storage/surreal-storage.js';
import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

describe('Schema initialization', () => {
  let tempDir: string;
  let storage: SurrealStorage;

  beforeEach(async () => {
    tempDir = mkdtempSync(join(tmpdir(), 'schema-test-'));
    storage = await SurrealStorage.create(join(tempDir, 'test.surql'));
  });

  afterEach(async () => {
    await storage.close();
    rmSync(tempDir, { recursive: true, force: true });
  });

  describe('schema version', () => {
    it('should report a schema version after initialization', async () => {
      const version = await storage.getSchemaVersion();
      expect(version).toBeGreaterThanOrEqual(1);
    });
  });

  describe('idempotent schema application', () => {
    it('should handle creating a second storage on the same path', async () => {
      const dbPath = join(tempDir, 'test.surql');
      await storage.close();

      // Re-open on the same path — schema should apply idempotently
      const storage2 = await SurrealStorage.create(dbPath);
      const version = await storage2.getSchemaVersion();
      expect(version).toBeGreaterThanOrEqual(1);

      // Reassign for afterEach cleanup
      storage = storage2;
    });
  });

  describe('table operations', () => {
    it('should support CRUD on semantic_concept table', async () => {
      await storage.insertSemanticConcept({
        id: 'sc-1',
        conceptName: 'TestConcept',
        conceptType: 'class',
        confidenceScore: 0.85,
        relationships: { uses: ['OtherClass'] },
        evolutionHistory: {},
        filePath: './src/test.ts',
        lineRange: { start: 1, end: 20 }
      });

      const concepts = await storage.getSemanticConcepts();
      expect(concepts).toHaveLength(1);
      expect(concepts[0].conceptName).toBe('TestConcept');
      expect(concepts[0].confidenceScore).toBe(0.85);
    });

    it('should support CRUD on developer_pattern table', async () => {
      await storage.insertDeveloperPattern({
        patternId: 'dp-1',
        patternType: 'naming_convention',
        patternContent: { description: 'Uses camelCase for variables' },
        frequency: 42,
        contexts: ['variables', 'parameters'],
        examples: [{ code: 'const myVar = 1;' }],
        confidence: 0.92
      });

      const patterns = await storage.getDeveloperPatterns();
      expect(patterns).toHaveLength(1);
      expect(patterns[0].patternType).toBe('naming_convention');
      expect(patterns[0].frequency).toBe(42);
    });

    it('should support file intelligence table', async () => {
      await storage.insertFileIntelligence({
        filePath: './src/main.ts',
        fileHash: 'abc123',
        semanticConcepts: ['EntryPoint', 'Server'],
        patternsUsed: ['async-main'],
        dependencies: ['./config.ts'],
        complexityMetrics: { cyclomatic: 5, cognitive: 3 },
        lastAnalyzed: new Date()
      });

      const intel = await storage.getFileIntelligence('./src/main.ts');
      expect(intel).toBeDefined();
      expect(intel!.filePath).toBe('./src/main.ts');
    });

    it('should support AI insights table', async () => {
      await storage.insertAIInsight({
        insightId: 'ai-1',
        insightType: 'architecture',
        insightContent: { summary: 'Service-oriented architecture detected' },
        confidenceScore: 0.88,
        sourceAgent: 'test',
        validationStatus: 'pending',
        impactPrediction: {}
      });

      const insights = await storage.getAIInsights();
      expect(insights).toHaveLength(1);
      expect(insights[0].insightType).toBe('architecture');
    });

    it('should support work session table', async () => {
      await storage.createWorkSession({
        id: 'ws-1',
        projectPath: '/tmp/project',
        currentFiles: [],
        completedTasks: [],
        pendingTasks: ['refactoring'],
        blockers: []
      });

      const session = await storage.getCurrentWorkSession('/tmp/project');
      expect(session).toBeDefined();
      expect(session!.pendingTasks).toContain('refactoring');
    });

    it('should support feature map table', async () => {
      await storage.insertFeatureMap({
        id: 'fm-1',
        projectPath: '/tmp/project',
        featureName: 'Authentication',
        primaryFiles: ['./src/auth.ts'],
        relatedFiles: ['./src/middleware.ts'],
        dependencies: ['jsonwebtoken'],
        status: 'active'
      });

      const features = await storage.getFeatureMaps('/tmp/project');
      expect(features).toHaveLength(1);
      expect(features[0].featureName).toBe('Authentication');
    });

    it('should support project decisions table', async () => {
      await storage.upsertProjectDecision({
        id: 'pd-1',
        projectPath: '/tmp/project',
        decisionKey: 'orm',
        decisionValue: 'prisma',
        reasoning: 'Type-safe queries'
      });

      const decision = await storage.getProjectDecision('/tmp/project', 'orm');
      expect(decision).toBeDefined();
      expect(decision!.decisionValue).toBe('prisma');
    });
  });

  describe('health check', () => {
    it('should pass health check on fresh database', async () => {
      await expect(storage.healthCheck()).resolves.not.toThrow();
    });
  });
});
