import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { SurrealStorage } from '../storage/surreal-storage.js';
import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

describe('SurrealStorage', () => {
  let tempDir: string;
  let storage: SurrealStorage;

  beforeEach(async () => {
    tempDir = mkdtempSync(join(tmpdir(), 'anime-mori-db-test-'));
    storage = await SurrealStorage.create(join(tempDir, 'test.surql'));
  });

  afterEach(async () => {
    await storage.close();
    rmSync(tempDir, { recursive: true, force: true });
  });

  it('should initialize storage with schema', async () => {
    expect(storage).toBeDefined();

    // Test that tables exist by trying to query them
    const concepts = await storage.getSemanticConcepts();
    expect(Array.isArray(concepts)).toBe(true);

    const patterns = await storage.getDeveloperPatterns();
    expect(Array.isArray(patterns)).toBe(true);
  });

  it('should report schema version', async () => {
    const version = await storage.getSchemaVersion();
    expect(version).toBeGreaterThanOrEqual(1);
  });

  it('should store and retrieve semantic concepts', async () => {
    const concept = {
      id: 'test-concept',
      conceptName: 'TestClass',
      conceptType: 'class',
      confidenceScore: 0.95,
      relationships: { extends: [] },
      evolutionHistory: { versions: [] },
      filePath: './test.ts',
      lineRange: { start: 1, end: 10 }
    };

    await storage.insertSemanticConcept(concept);

    const stored = await storage.getSemanticConcepts();
    expect(stored.length).toBe(1);
    expect(stored[0].conceptName).toBe('TestClass');
  });

  it('should filter semantic concepts by file path', async () => {
    await storage.insertSemanticConcept({
      id: 'concept-a',
      conceptName: 'ClassA',
      conceptType: 'class',
      confidenceScore: 0.9,
      relationships: {},
      evolutionHistory: {},
      filePath: './a.ts',
      lineRange: { start: 1, end: 5 }
    });

    await storage.insertSemanticConcept({
      id: 'concept-b',
      conceptName: 'ClassB',
      conceptType: 'class',
      confidenceScore: 0.8,
      relationships: {},
      evolutionHistory: {},
      filePath: './b.ts',
      lineRange: { start: 1, end: 5 }
    });

    const all = await storage.getSemanticConcepts();
    expect(all.length).toBe(2);

    const filtered = await storage.getSemanticConcepts('./a.ts');
    expect(filtered.length).toBe(1);
    expect(filtered[0].conceptName).toBe('ClassA');
  });

  it('should store and retrieve developer patterns', async () => {
    await storage.insertDeveloperPattern({
      patternId: 'pattern-1',
      patternType: 'naming',
      patternContent: { description: 'camelCase naming' },
      frequency: 10,
      contexts: ['variables', 'functions'],
      examples: [{ code: 'const myVar = 1;' }],
      confidence: 0.9
    });

    const patterns = await storage.getDeveloperPatterns();
    expect(patterns.length).toBe(1);
    expect(patterns[0].patternType).toBe('naming');
  });

  it('should filter developer patterns by type', async () => {
    await storage.insertDeveloperPattern({
      patternId: 'p1',
      patternType: 'naming',
      patternContent: {},
      frequency: 5,
      contexts: [],
      examples: [],
      confidence: 0.8
    });

    await storage.insertDeveloperPattern({
      patternId: 'p2',
      patternType: 'structure',
      patternContent: {},
      frequency: 3,
      contexts: [],
      examples: [],
      confidence: 0.7
    });

    const naming = await storage.getDeveloperPatterns('naming');
    expect(naming.length).toBe(1);
    expect(naming[0].patternId).toBe('p1');
  });

  it('should store and retrieve project metadata', async () => {
    await storage.insertProjectMetadata({
      projectId: 'proj-1',
      projectPath: '/tmp/test',
      projectName: 'test-project',
      languagePrimary: 'typescript',
      languagesDetected: ['typescript', 'javascript'],
      frameworkDetected: ['node'],
      intelligenceVersion: '0.6.0',
      lastFullScan: new Date()
    });

    const metadata = await storage.getProjectMetadata('/tmp/test');
    expect(metadata).toBeDefined();
    expect(metadata!.projectName).toBe('test-project');
  });

  it('should pass health check', async () => {
    await expect(storage.healthCheck()).resolves.not.toThrow();
  });

  it('should handle close gracefully', async () => {
    await expect(storage.close()).resolves.not.toThrow();
    // Create a new storage for the afterEach cleanup
    storage = await SurrealStorage.create(join(tempDir, 'test.surql'));
  });
});
