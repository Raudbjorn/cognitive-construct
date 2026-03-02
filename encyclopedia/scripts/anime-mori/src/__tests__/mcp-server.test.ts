import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { CodeCartographerMCP } from '../mcp-server/server.js';
import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

describe('CodeCartographerMCP Server', () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'anime-mori-mcp-test-'));
    // Set test database path
    process.env.ANIME_MORI_DB_PATH = join(tempDir, 'test.surql');
  });

  afterEach(() => {
    rmSync(tempDir, { recursive: true, force: true });
    delete process.env.ANIME_MORI_DB_PATH;
  });

  it('should create MCP server instance', () => {
    const server = new CodeCartographerMCP();
    expect(server).toBeDefined();
  });

  it('should handle tool routing without errors', async () => {
    const server = new CodeCartographerMCP();
    await server.initializeForTesting();

    // Test invalid tool name — should throw with 'Unknown tool'
    try {
      await server.routeToolCall('invalid_tool', {});
      expect.fail('Should have thrown error for invalid tool');
    } catch (error: unknown) {
      expect((error as Error).message).toContain('Unknown tool');
    }

    await server.stop();
  });
});