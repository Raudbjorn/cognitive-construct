#!/usr/bin/env node

/**
 * Anime Mori CLI — wraps MCP server tools as subcommands.
 *
 * Exit codes:
 *   0 = success
 *   1 = runtime error
 *   2 = usage error
 *   3 = initialization error
 */

import { initializeComponents, shutdownComponents } from './cli/init.js';
import { routeToolCall } from './cli/router.js';
import { runServer } from './mcp-server/server.js';
import { InteractiveSetup } from './cli/interactive-setup.js';
import { DebugTools } from './cli/debug-tools.js';

// ── Subcommand → MCP tool mapping ──────────────────────────────────────────

interface SubcommandDef {
  tool: string;
  description: string;
  positional?: string;       // name of the positional arg field
  flags: Record<string, FlagDef>;
}

interface FlagDef {
  field: string;             // tool arg field name
  type: 'boolean' | 'string' | 'number' | 'json';
  default?: any;
  description: string;
  negate?: boolean;          // --no-<flag> sets field to false
}

const SUBCOMMANDS: Record<string, SubcommandDef> = {
  analyze: {
    tool: 'analyze_codebase',
    description: 'Analyze a specific file or directory',
    positional: 'path',
    flags: {
      '--include-content': { field: 'includeContent', type: 'boolean', default: false, description: 'Include file contents in output' },
    },
  },
  search: {
    tool: 'search_codebase',
    description: 'Search the codebase',
    positional: 'query',
    flags: {
      '--type': { field: 'type', type: 'string', description: 'Search type: semantic|text|pattern' },
      '--language': { field: 'language', type: 'string', description: 'Filter by language' },
      '--limit': { field: 'limit', type: 'number', description: 'Max results (default: 20)' },
    },
  },
  learn: {
    tool: 'learn_codebase_intelligence',
    description: 'Build intelligence database from codebase',
    positional: 'path',
    flags: {
      '--force': { field: 'force', type: 'boolean', default: false, description: 'Force re-learning' },
    },
  },
  'auto-learn': {
    tool: 'auto_learn_if_needed',
    description: 'Learn from codebase if intelligence data is missing or stale',
    positional: 'path',
    flags: {
      '--force': { field: 'force', type: 'boolean', default: false, description: 'Force re-learning' },
      '--skip-learning': { field: 'skipLearning', type: 'boolean', default: false, description: 'Check status only, do not learn' },
    },
  },
  insights: {
    tool: 'get_semantic_insights',
    description: 'Get semantic insights from learned intelligence',
    positional: 'query',
    flags: {
      '--type': { field: 'conceptType', type: 'string', description: 'Concept type filter' },
      '--limit': { field: 'limit', type: 'number', description: 'Max results (default: 10)' },
    },
  },
  patterns: {
    tool: 'get_pattern_recommendations',
    description: 'Get pattern recommendations for a problem',
    positional: 'problemDescription',
    flags: {
      '--file': { field: 'currentFile', type: 'string', description: 'Current file path for context' },
      '--code': { field: 'selectedCode', type: 'string', description: 'Selected code snippet' },
    },
  },
  predict: {
    tool: 'predict_coding_approach',
    description: 'Predict coding approach for a problem',
    positional: 'problemDescription',
    flags: {
      '--context': { field: 'context', type: 'json', description: 'Additional context as JSON' },
    },
  },
  profile: {
    tool: 'get_developer_profile',
    description: 'Get developer profile from learned patterns',
    flags: {
      '--no-recent': { field: 'includeRecentActivity', type: 'boolean', negate: true, description: 'Exclude recent activity' },
    },
  },
  contribute: {
    tool: 'contribute_insights',
    description: 'Contribute insights to the intelligence database',
    flags: {
      '--type': { field: 'type', type: 'string', description: 'Insight type: bug_pattern|optimization|refactor_suggestion|best_practice' },
      '--content': { field: 'content', type: 'json', description: 'Insight content as JSON' },
      '--confidence': { field: 'confidence', type: 'number', description: 'Confidence score (0-1)' },
      '--source': { field: 'sourceAgent', type: 'string', description: 'Source agent identifier' },
    },
  },
  blueprint: {
    tool: 'get_project_blueprint',
    description: 'Get project blueprint with architecture overview',
    positional: 'path',
    flags: {
      '--no-feature-map': { field: 'includeFeatureMap', type: 'boolean', negate: true, description: 'Exclude feature map' },
    },
  },
  status: {
    tool: 'get_system_status',
    description: 'Get system status and health indicators',
    flags: {
      '--no-metrics': { field: 'includeMetrics', type: 'boolean', negate: true, description: 'Exclude performance metrics' },
      '--diagnostics': { field: 'includeDiagnostics', type: 'boolean', default: false, description: 'Include system diagnostics' },
    },
  },
  metrics: {
    tool: 'get_intelligence_metrics',
    description: 'Get intelligence metrics and statistics',
    flags: {
      '--no-breakdown': { field: 'includeBreakdown', type: 'boolean', negate: true, description: 'Exclude detailed breakdown' },
    },
  },
  perf: {
    tool: 'get_performance_status',
    description: 'Get performance status',
    flags: {
      '--benchmark': { field: 'runBenchmark', type: 'boolean', default: false, description: 'Run performance benchmark' },
    },
  },
  health: {
    tool: 'health_check',
    description: 'Run health check',
    positional: 'path',
    flags: {},
  },
};

// ── Arg parser ──────────────────────────────────────────────────────────────

interface ParsedArgs {
  command: string;
  toolArgs: Record<string, any>;
  raw: string[];
}

function parseArgs(argv: string[]): ParsedArgs {
  const args = argv.slice(2); // strip node + script
  const command = args[0] || '--help';

  if (command === '--help' || command === '-h') {
    printHelp();
    process.exit(0);
  }

  if (command === '--version' || command === '-v') {
    console.log('0.6.0');
    process.exit(0);
  }

  // Special commands without tool routing
  if (['server', 'setup', 'debug'].includes(command)) {
    return { command, toolArgs: {}, raw: args.slice(1) };
  }

  const def = SUBCOMMANDS[command];
  if (!def) {
    console.error(`Unknown command: ${command}\n`);
    printHelp();
    process.exit(2);
  }

  const toolArgs: Record<string, any> = {};
  const rest = args.slice(1);
  let positionalConsumed = false;

  for (let i = 0; i < rest.length; i++) {
    const arg = rest[i];

    if (arg === '--help' || arg === '-h') {
      printCommandHelp(command, def);
      process.exit(0);
    }

    // Check for flags
    if (arg.startsWith('--')) {
      // Handle --flag=value syntax
      const eqIdx = arg.indexOf('=');
      const flagName = eqIdx > -1 ? arg.slice(0, eqIdx) : arg;
      const flagDef = def.flags[flagName];

      if (!flagDef) {
        console.error(`Unknown flag for '${command}': ${flagName}`);
        process.exit(2);
      }

      let value: any;
      if (flagDef.negate) {
        value = false;
      } else if (flagDef.type === 'boolean') {
        value = true;
      } else if (eqIdx > -1) {
        value = arg.slice(eqIdx + 1);
      } else {
        i++;
        if (i >= rest.length) {
          console.error(`Flag '${flagName}' requires a value`);
          process.exit(2);
        }
        value = rest[i];
      }

      // Coerce types
      if (flagDef.type === 'number') {
        value = Number(value);
        if (Number.isNaN(value)) {
          console.error(`Flag '${flagName}' requires a numeric value`);
          process.exit(2);
        }
      } else if (flagDef.type === 'json') {
        try {
          value = JSON.parse(value);
        } catch {
          console.error(`Flag '${flagName}' requires valid JSON`);
          process.exit(2);
        }
      }

      toolArgs[flagDef.field] = value;
    } else if (def.positional && !positionalConsumed) {
      // First non-flag argument is the positional
      toolArgs[def.positional] = arg;
      positionalConsumed = true;
    } else {
      console.error(`Unexpected argument: ${arg}`);
      process.exit(2);
    }
  }

  return { command, toolArgs, raw: rest };
}

// ── Help text ───────────────────────────────────────────────────────────────

function printHelp(): void {
  const lines = [
    'Anime Mori — Persistent codebase intelligence for AI assistants',
    '',
    'Usage: anime-mori <command> [args] [flags]',
    '',
    'Commands:',
  ];

  // Tool commands
  const entries = Object.entries(SUBCOMMANDS).map(([name, def]) => {
    const pos = def.positional ? ` <${def.positional}>` : '';
    return { label: name + pos, desc: def.description };
  });
  const specials = [
    { label: 'server', desc: 'Start MCP stdio transport server' },
    { label: 'setup', desc: 'Interactive setup wizard' },
    { label: 'debug [path]', desc: 'Run diagnostics' },
  ];
  const colWidth = Math.max(...[...entries, ...specials].map(e => e.label.length)) + 4;

  for (const { label, desc } of entries) {
    lines.push(`  ${label.padEnd(colWidth)}${desc}`);
  }

  // Special commands
  lines.push('');
  for (const { label, desc } of specials) {
    lines.push(`  ${label.padEnd(colWidth)}${desc}`);
  }
  lines.push('');
  lines.push('Options:');
  lines.push('  --help, -h       Show this help');
  lines.push('  --version, -v    Show version');
  lines.push('');
  lines.push('Environment:');
  lines.push('  ANIME_MORI_STORAGE_DIR      Database storage location (default: project dir)');
  lines.push('  ANIME_MORI_DB_FILENAME      Database filename (default: anime-mori.surql)');
  lines.push('  ANIME_MORI_LOG_LEVEL        error|warn|info|debug (default: info)');
  lines.push('');
  lines.push('Run `anime-mori <command> --help` for command-specific flags.');

  console.error(lines.join('\n'));
}

function printCommandHelp(command: string, def: SubcommandDef): void {
  const pos = def.positional ? ` <${def.positional}>` : '';
  const lines = [
    `Usage: anime-mori ${command}${pos} [flags]`,
    '',
    def.description,
    '',
  ];

  const flagEntries = Object.entries(def.flags);
  if (flagEntries.length > 0) {
    lines.push('Flags:');
    const maxLen = Math.max(...flagEntries.map(([k]) => k.length));
    for (const [name, flag] of flagEntries) {
      lines.push(`  ${name.padEnd(maxLen + 4)}${flag.description}`);
    }
  }

  console.error(lines.join('\n'));
}

// ── Main ────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const { command, toolArgs, raw } = parseArgs(process.argv);

  // Special command: MCP server mode
  if (command === 'server') {
    await runServer();
    return;
  }

  // Special command: interactive setup
  if (command === 'setup') {
    const setup = new InteractiveSetup();
    await setup.run();
    return;
  }

  // Special command: debug diagnostics
  if (command === 'debug') {
    const projectPath = raw[0] || process.cwd();
    const verbose = raw.includes('--verbose');
    const debugTools = new DebugTools({
      verbose,
      checkDatabase: true,
      checkIntelligence: true,
      checkFileSystem: true,
      validateData: verbose,
      performance: verbose,
    });
    await debugTools.runDiagnostics(projectPath);
    return;
  }

  // Tool commands: force Logger to stderr so stdout is clean JSON
  process.env.MCP_SERVER = 'true';

  const components = await initializeComponents().catch((error) => {
    console.error(`Initialization failed: ${error instanceof Error ? error.message : String(error)}`);
    process.exit(3);
  });

  try {
    const def = SUBCOMMANDS[command];
    const result = await routeToolCall(def.tool, toolArgs, components);
    // Clean JSON output to stdout
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    console.error(`Error: ${error instanceof Error ? error.message : String(error)}`);
    process.exit(1);
  } finally {
    await shutdownComponents(components);
  }
}

main().catch((error) => {
  console.error(`Fatal: ${error instanceof Error ? error.message : String(error)}`);
  process.exit(1);
});
