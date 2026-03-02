/**
 * Standalone tool routing for both MCP server and CLI modes.
 * Extracted from CodeCartographerMCP.routeToolCall()
 *
 * Throws plain Error (not McpError) — callers wrap as needed.
 */

import { z } from 'zod';
import { VALIDATION_SCHEMAS } from '../mcp-server/validation.js';
import type { AnimeMoriComponents } from './init.js';

/**
 * Validate input against a Zod schema, throwing a plain Error on failure.
 */
function validateInput(schema: z.ZodSchema<any>, input: any, toolName: string): any {
  try {
    return schema.parse(input);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const messages = error.issues.map((e: z.ZodIssue) => `${e.path.join('.')}: ${e.message}`).join(', ');
      throw new Error(`Invalid input for ${toolName}: ${messages}`);
    }
    throw new Error(`Validation error for ${toolName}: ${(error as Error).message}`);
  }
}

/**
 * Route a tool call to the appropriate handler.
 * Returns the raw result object (caller is responsible for serialization).
 */
export async function routeToolCall(
  name: string,
  args: any,
  components: AnimeMoriComponents
): Promise<any> {
  // Validate input using Zod schemas
  const schema = VALIDATION_SCHEMAS[name as keyof typeof VALIDATION_SCHEMAS];
  if (schema) {
    args = validateInput(schema, args, name);
  }

  const { coreTools, intelligenceTools, automationTools, monitoringTools } = components;

  switch (name) {
    // Core Analysis Tools
    case 'analyze_codebase':
      return await coreTools.analyzeCodebase(args);

    case 'search_codebase':
      return await coreTools.searchCodebase(args);

    // Intelligence Tools
    case 'learn_codebase_intelligence':
      return await intelligenceTools.learnCodebaseIntelligence(args);

    case 'get_semantic_insights':
      return await intelligenceTools.getSemanticInsights(args);

    case 'get_pattern_recommendations':
      return await intelligenceTools.getPatternRecommendations(args);

    case 'predict_coding_approach':
      return await intelligenceTools.predictCodingApproach(args);

    case 'get_developer_profile':
      return await intelligenceTools.getDeveloperProfile(args);

    case 'contribute_insights':
      return await intelligenceTools.contributeInsights(args);

    case 'get_project_blueprint':
      return await intelligenceTools.getProjectBlueprint(args);

    // Automation Tools
    case 'auto_learn_if_needed':
      return await automationTools.autoLearnIfNeeded(args);

    // Monitoring Tools
    case 'get_system_status':
      return await monitoringTools.getSystemStatus(args);

    case 'get_intelligence_metrics':
      return await monitoringTools.getIntelligenceMetrics(args);

    case 'get_performance_status':
      return await monitoringTools.getPerformanceStatus(args);

    case 'health_check':
      return await monitoringTools.healthCheck(args);

    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

/** All known tool names, for help text and validation */
export const TOOL_NAMES = Object.keys(VALIDATION_SCHEMAS);
