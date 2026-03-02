/**
 * Global test setup file
 * This file runs before all tests
 */

import { beforeAll, afterAll, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, rmSync, existsSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

// Global test state
let globalTempDir: string | null = null;

/**
 * Setup before all tests
 */
beforeAll(() => {
    console.log('🧪 Starting Anime-Mori test suite...\n');

    // Create global temp directory for test artifacts
    globalTempDir = mkdtempSync(join(tmpdir(), 'anime-mori-tests-'));

    // Set test environment variables
    process.env.NODE_ENV = 'test';
    process.env.ANIME_MORI_TEST_MODE = 'true';
});

/**
 * Cleanup after all tests
 */
afterAll(() => {
    console.log('\n✅ Test suite completed\n');

    // Cleanup global temp directory
    if (globalTempDir && existsSync(globalTempDir)) {
        try {
            rmSync(globalTempDir, { recursive: true, force: true });
        } catch (error) {
            console.warn('Failed to cleanup global temp directory:', error);
        }
    }

    // Cleanup environment variables
    delete process.env.ANIME_MORI_DB_PATH;
    delete process.env.ANIME_MORI_TEST_MODE;
});

/**
 * Setup before each test
 */
beforeEach(() => {
    // Reset any global state if needed
});

/**
 * Cleanup after each test
 */
afterEach(() => {
    // Cleanup test-specific resources
    delete process.env.ANIME_MORI_DB_PATH;
});

// Export utilities for tests
export function createTestTempDir(prefix: string = 'test-'): string {
    return mkdtempSync(join(tmpdir(), `anime-mori-${prefix}-`));
}

export function cleanupTestTempDir(dir: string): void {
    if (existsSync(dir)) {
        rmSync(dir, { recursive: true, force: true });
    }
}

export function getGlobalTempDir(): string {
    if (!globalTempDir) {
        throw new Error('Global temp directory not initialized');
    }
    return globalTempDir;
}
