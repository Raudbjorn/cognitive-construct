import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { SemanticEngine } from '../engines/semantic-engine.js';
import { SurrealStorage } from '../storage/surreal-storage.js';
import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

describe('SemanticEngine', () => {
  let tempDir: string;
  let storage: SurrealStorage;
  let semanticEngine: SemanticEngine;

  beforeEach(async () => {
    tempDir = mkdtempSync(join(tmpdir(), 'anime-mori-test-'));
    storage = await SurrealStorage.create(join(tempDir, 'test.surql'));
    semanticEngine = new SemanticEngine(storage);
  });

  afterEach(async () => {
    await storage.close();
    rmSync(tempDir, { recursive: true, force: true });
  });

  it('should initialize without errors', () => {
    expect(semanticEngine).toBeDefined();
  });

  it('should analyze codebase structure', async () => {
    const testCodePath = './src';
    const analysis = await semanticEngine.analyzeCodebase(testCodePath);

    expect(analysis).toBeDefined();
    expect(Array.isArray(analysis.languages)).toBe(true);
    expect(Array.isArray(analysis.concepts)).toBe(true);
    expect(typeof analysis.complexity).toBe('object');
  });

  it('should analyze file content and extract concepts', async () => {
    const sampleCode = `
      export class TestClass {
        private value: number;

        constructor(value: number) {
          this.value = value;
        }

        getValue(): number {
          return this.value;
        }
      }
    `;

    const concepts = await semanticEngine.analyzeFileContent('./test.ts', sampleCode);
    expect(Array.isArray(concepts)).toBe(true);
  });
});
