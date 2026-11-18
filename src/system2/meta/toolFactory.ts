import * as fs from 'fs';
import * as path from 'path';
import { MCPToolSchema } from '../../types';
import { CONFIG } from '../../config';
import { DynamicMCPToolRegistry } from '../mcp/toolRegistry';
import { UnifiedLLMClient } from '../llmClient';
import { validateAndCompileSynthesizedTool } from './sandbox';

interface PersistedSynthesizedTool {
  schema: MCPToolSchema;
  functionBodyCode: string;
}

/**
 * Self-Synthesizing MCP Tool Factory.
 * Dynamically writes, validates in a VM sandbox, persists to disk, and hot-reloads
 * new MCP tools whenever the High-Level Meta-Agent detects a novel user command.
 */
export class SelfSynthesizingToolFactory {
  constructor(
    private readonly registry: DynamicMCPToolRegistry,
    private readonly llm: UnifiedLLMClient,
    private readonly registryFilePath = CONFIG.paths.skillRegistryFile,
    private readonly skillsDir = CONFIG.paths.synthesizedSkillsDir
  ) {
    this.loadPersistedSynthesizedTools();
  }

  /**
   * Synthesizes a new MCP tool, validates it in the AST/VM sandbox, persists it,
   * and registers it into the active MCP catalog (emitting `notifications/tools/list_changed`).
   */
  public async synthesizeAndRegisterTool(
    desiredToolName: string,
    taskDescription: string,
    requiredParameters: Record<string, string>,
    triggeringAsk: string
  ): Promise<{ success: boolean; toolName: string; error?: string }> {
    const cleanName = desiredToolName
      .toLowerCase()
      .replace(/[^a-z0-9_]/g, '_')
      .replace(/^_+|_+$/g, '')
      .slice(0, 40);

    const safeToolName = /^[a-z]/.test(cleanName) ? cleanName : `skill_${cleanName || 'custom'}`;

    console.log(`[ToolFactory] Synthesizing new MCP tool "${safeToolName}" for ask: "${triggeringAsk}"...`);

    const generated = await this.llm.synthesizeToolCode(
      safeToolName,
      taskDescription,
      requiredParameters,
      triggeringAsk
    );

    const validation = validateAndCompileSynthesizedTool(safeToolName, generated.functionBodyCode);
    if (!validation.valid || !validation.compiledHandler) {
      console.warn(`[ToolFactory] Sandbox rejected synthesized tool "${safeToolName}": ${validation.error}`);
      return {
        success: false,
        toolName: safeToolName,
        error: validation.error,
      };
    }

    const schema: MCPToolSchema = {
      name: safeToolName,
      description: generated.description,
      inputSchema: {
        type: 'object',
        properties: generated.parameters,
      },
      isSynthesized: true,
      synthesizedAt: new Date().toISOString(),
      sourceCode: generated.functionBodyCode,
      tags: generated.tags,
    };

    // Register into live MCP Registry (automatically emits notifications/tools/list_changed)
    this.registry.registerTool(schema, validation.compiledHandler);

    // Persist to disk so the agent permanently remembers this skill across restarts
    this.persistToolToDisk(schema, generated.functionBodyCode);

    console.log(`[ToolFactory] Hot-reloaded new MCP tool "${safeToolName}" into active catalog!`);
    return {
      success: true,
      toolName: safeToolName,
    };
  }

  private persistToolToDisk(schema: MCPToolSchema, functionBodyCode: string): void {
    try {
      if (!fs.existsSync(this.skillsDir)) {
        fs.mkdirSync(this.skillsDir, { recursive: true });
      }

      // Write human-readable skill file in data/synthesized_skills/<name>.js
      const jsFile = path.join(this.skillsDir, `${schema.name}.js`);
      const fileHeader = `// Synthesized MCP Tool: ${schema.name}\n// Created: ${schema.synthesizedAt}\n// Description: ${schema.description}\n\n`;
      fs.writeFileSync(jsFile, fileHeader + functionBodyCode.trim() + '\n', 'utf8');

      // Update JSON index
      let existing: PersistedSynthesizedTool[] = [];
      if (fs.existsSync(this.registryFilePath)) {
        existing = JSON.parse(fs.readFileSync(this.registryFilePath, 'utf8'));
      }
      existing = existing.filter((e) => e.schema.name !== schema.name);
      existing.push({ schema, functionBodyCode });

      const dir = path.dirname(this.registryFilePath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      fs.writeFileSync(this.registryFilePath, JSON.stringify(existing, null, 2), 'utf8');
    } catch (err: any) {
      console.warn(`[ToolFactory] Could not persist synthesized tool: ${err.message}`);
    }
  }

  private loadPersistedSynthesizedTools(): void {
    try {
      if (!fs.existsSync(this.registryFilePath)) return;
      const raw = fs.readFileSync(this.registryFilePath, 'utf8');
      const items: PersistedSynthesizedTool[] = JSON.parse(raw);

      for (const item of items) {
        const res = validateAndCompileSynthesizedTool(item.schema.name, item.functionBodyCode);
        if (res.valid && res.compiledHandler) {
          this.registry.registerTool(item.schema, res.compiledHandler);
        }
      }

      if (items.length > 0) {
        console.log(`[ToolFactory] Restored ${items.length} previously synthesized MCP tool(s) from disk.`);
      }
    } catch {
      // Ignore corrupt or empty file
    }
  }
}
