import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from '@modelcontextprotocol/sdk/types.js';

import { initializeComponents, shutdownComponents, type AnimeMoriComponents } from '../cli/init.js';
import { routeToolCall } from '../cli/router.js';
import { Logger } from '../utils/logger.js';

export class CodeCartographerMCP {
  private server: Server;
  private components!: AnimeMoriComponents;

  constructor() {
    this.server = new Server(
      {
        name: 'anime-mori',
        version: '0.6.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupHandlers();
  }

  private setupHandlers(): void {
    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          ...this.components.coreTools.tools,
          ...this.components.intelligenceTools.tools,
          ...this.components.automationTools.tools,
          ...this.components.monitoringTools.tools
        ]
      };
    });

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        const result = await this.routeToolCall(name, args);

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(result, null, 2)
            }
          ]
        };
      } catch (error) {
        if (error instanceof McpError) {
          throw error;
        }

        throw new McpError(
          ErrorCode.InternalError,
          `Tool execution failed: ${error instanceof Error ? error.message : String(error)}`
        );
      }
    });
  }

  /**
   * Route a tool call — delegates to the shared router, wrapping errors in McpError.
   */
  public async routeToolCall(name: string, args: any): Promise<any> {
    try {
      return await routeToolCall(name, args, this.components);
    } catch (error) {
      if (error instanceof McpError) {
        throw error;
      }
      // Wrap plain errors from the shared router as McpError
      const message = error instanceof Error ? error.message : String(error);
      if (message.startsWith('Unknown tool:')) {
        throw new McpError(ErrorCode.MethodNotFound, message);
      }
      if (message.startsWith('Invalid input for')) {
        throw new McpError(ErrorCode.InvalidParams, message);
      }
      throw new McpError(ErrorCode.InternalError, message);
    }
  }

  async start(): Promise<void> {
    // Set environment variable to indicate MCP server mode
    process.env.MCP_SERVER = 'true';

    this.components = await initializeComponents();

    const transport = new StdioServerTransport();
    await this.server.connect(transport);

    Logger.info('Anime Mori MCP Server started');
  }

  /**
   * Get all registered tools (for testing and introspection)
   */
  getAllTools(): any[] {
    return [
      ...this.components.coreTools.tools,
      ...this.components.intelligenceTools.tools,
      ...this.components.automationTools.tools,
      ...this.components.monitoringTools.tools
    ];
  }

  /**
   * Initialize components for testing without starting transport
   */
  async initializeForTesting(): Promise<void> {
    this.components = await initializeComponents();
  }

  async stop(): Promise<void> {
    if (this.components) {
      await shutdownComponents(this.components);
    }

    // Close MCP server
    await this.server.close();
  }
}

// Export for CLI usage
export async function runServer(): Promise<void> {
  const server = new CodeCartographerMCP();

  // Handle graceful shutdown
  process.on('SIGINT', async () => {
    await server.stop();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    await server.stop();
    process.exit(0);
  });

  await server.start();
}
