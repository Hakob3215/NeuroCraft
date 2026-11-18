import { MCPToolExecutionContext, MCPToolSchema } from '../../types';
import { DynamicMCPToolRegistry } from './toolRegistry';

export interface JsonRpcRequest {
  jsonrpc: '2.0';
  id?: string | number;
  method: string;
  params?: Record<string, any>;
}

export interface JsonRpcResponse {
  jsonrpc: '2.0';
  id?: string | number;
  result?: any;
  error?: { code: number; message: string; data?: any };
}

/**
 * In-process and JSON-RPC 2.0 compliant Model Context Protocol (MCP) Server.
 * Exposes `tools/list`, `tools/call`, and emits `notifications/tools/list_changed`
 * whenever the Self-Synthesizing Tool Factory creates a new tool at runtime.
 */
export class NeuroCraftMCPServer {
  private notificationListeners: Array<(notification: { method: string; params: any }) => void> = [];

  constructor(public readonly registry: DynamicMCPToolRegistry) {
    this.registry.on('notifications/tools/list_changed', (info) => {
      this.broadcastNotification('notifications/tools/list_changed', info);
    });
  }

  public onNotification(listener: (notification: { method: string; params: any }) => void): void {
    this.notificationListeners.push(listener);
  }

  private broadcastNotification(method: string, params: any): void {
    for (const listener of this.notificationListeners) {
      listener({ method, params });
    }
  }

  /**
   * Handles standard MCP JSON-RPC 2.0 requests (`initialize`, `tools/list`, `tools/call`).
   */
  public async handleRpcRequest(
    request: JsonRpcRequest,
    executionContextFactory: (toolName: string, args: Record<string, any>) => MCPToolExecutionContext
  ): Promise<JsonRpcResponse> {
    const { id, method, params } = request;

    try {
      if (method === 'initialize') {
        return {
          jsonrpc: '2.0',
          id,
          result: {
            protocolVersion: '2024-11-05',
            serverInfo: {
              name: 'neurocraft-dynamic-mcp',
              version: '1.0.0',
            },
            capabilities: {
              tools: { listChanged: true },
            },
          },
        };
      }

      if (method === 'tools/list') {
        const tools: MCPToolSchema[] = this.registry.listSchemas();
        return {
          jsonrpc: '2.0',
          id,
          result: { tools },
        };
      }

      if (method === 'tools/call') {
        const toolName = params?.name;
        const toolArgs = params?.arguments || {};

        const tool = this.registry.getTool(toolName);
        if (!tool) {
          return {
            jsonrpc: '2.0',
            id,
            error: {
              code: -32601,
              message: `MCP Tool "${toolName}" is not registered in catalog.`,
            },
          };
        }

        const normalizedArgs: Record<string, any> = { ...toolArgs };
        if (
          !normalizedArgs.itemName &&
          (normalizedArgs.recipeName || normalizedArgs.item || normalizedArgs.target || normalizedArgs.armor)
        ) {
          normalizedArgs.itemName =
            normalizedArgs.recipeName || normalizedArgs.item || normalizedArgs.target || normalizedArgs.armor;
        }
        if (!normalizedArgs.blockType && (normalizedArgs.item || normalizedArgs.target || normalizedArgs.block)) {
          normalizedArgs.blockType = normalizedArgs.item || normalizedArgs.target || normalizedArgs.block;
        }
        if (!normalizedArgs.targetMob && (normalizedArgs.target || normalizedArgs.mob || normalizedArgs.entity)) {
          normalizedArgs.targetMob = normalizedArgs.target || normalizedArgs.mob || normalizedArgs.entity;
        }

        const ctx = executionContextFactory(tool.schema.name, normalizedArgs);
        const output = await tool.handler(normalizedArgs, ctx);

        return {
          jsonrpc: '2.0',
          id,
          result: {
            content: [
              {
                type: 'text',
                text: JSON.stringify(output),
              },
            ],
            isError: !output.success,
          },
        };
      }

      return {
        jsonrpc: '2.0',
        id,
        error: {
          code: -32601,
          message: `Unsupported MCP method: ${method}`,
        },
      };
    } catch (err: any) {
      return {
        jsonrpc: '2.0',
        id,
        error: {
          code: -32000,
          message: err?.message || 'Internal MCP tool execution error',
        },
      };
    }
  }
}
