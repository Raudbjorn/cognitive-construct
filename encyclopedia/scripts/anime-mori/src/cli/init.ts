/**
 * Shared component initialization for both MCP server and CLI modes.
 * Extracted from CodeCartographerMCP.initializeComponents()
 */

import { CoreAnalysisTools } from '../mcp-server/tools/core-analysis.js';
import { IntelligenceTools } from '../mcp-server/tools/intelligence-tools.js';
import { AutomationTools } from '../mcp-server/tools/automation-tools.js';
import { MonitoringTools } from '../mcp-server/tools/monitoring-tools.js';
import { SemanticEngine } from '../engines/semantic-engine.js';
import { PatternEngine } from '../engines/pattern-engine.js';
import { SurrealStorage } from '../storage/surreal-storage.js';
import { config } from '../config/config.js';
import { Logger } from '../utils/logger.js';

export interface AnimeMoriComponents {
  storage: SurrealStorage;
  semanticEngine: SemanticEngine;
  patternEngine: PatternEngine;
  coreTools: CoreAnalysisTools;
  intelligenceTools: IntelligenceTools;
  automationTools: AutomationTools;
  monitoringTools: MonitoringTools;
  dbPath: string;
}

export async function initializeComponents(): Promise<AnimeMoriComponents> {
  Logger.info('Initializing Anime Mori components...');

  const dbPath = config.getDatabasePath();
  Logger.info(`Attempting to initialize database at: ${dbPath}`);

  let storage: SurrealStorage;
  try {
    storage = await SurrealStorage.create(dbPath);
    Logger.info('SurrealDB storage initialized successfully');
  } catch (dbError: unknown) {
    Logger.error('Failed to initialize SurrealDB storage:', dbError);
    throw new Error(
      `Storage initialization failed: ${dbError instanceof Error ? dbError.message : String(dbError)}`
    );
  }

  const semanticEngine = new SemanticEngine(storage);
  const patternEngine = new PatternEngine(storage);
  Logger.info('Analysis engines initialized');

  const coreTools = new CoreAnalysisTools(semanticEngine, patternEngine, storage);
  const intelligenceTools = new IntelligenceTools(semanticEngine, patternEngine, storage);
  const automationTools = new AutomationTools(semanticEngine, patternEngine, storage);
  const monitoringTools = new MonitoringTools(semanticEngine, patternEngine, storage, dbPath);
  Logger.info('Tool collections initialized');

  Logger.info('Anime Mori components initialized successfully');

  return {
    storage,
    semanticEngine,
    patternEngine,
    coreTools,
    intelligenceTools,
    automationTools,
    monitoringTools,
    dbPath,
  };
}

export async function shutdownComponents(components: AnimeMoriComponents): Promise<void> {
  if (components.semanticEngine) {
    components.semanticEngine.cleanup();
  }

  if (components.storage) {
    try {
      await components.storage.close();
    } catch (error) {
      Logger.warn('Failed to close storage:', error);
    }
  }
}
