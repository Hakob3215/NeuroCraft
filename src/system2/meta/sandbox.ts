import * as vm from 'vm';
import { Vec3 } from 'vec3';
import { MCPToolExecutionContext, MCPToolHandler } from '../../types';

export interface SandboxValidationResult {
  valid: boolean;
  error?: string;
  compiledHandler?: MCPToolHandler;
}

const FORBIDDEN_PATTERNS: Array<{ pattern: RegExp; reason: string }> = [
  { pattern: /\bchild_process\b/, reason: 'Access to child_process is forbidden in synthesized tools.' },
  { pattern: /\bprocess\.exit\b/, reason: 'Calling process.exit is forbidden.' },
  { pattern: /\brequire\s*\(/, reason: 'Dynamic require() is forbidden inside synthesized tool bodies.' },
  { pattern: /\bfs\.(rm|unlink|rmdir|writeFile)/, reason: 'Direct filesystem mutation is forbidden.' },
  { pattern: /\bwhile\s*\(\s*true\s*\)(?![\s\S]*?(?:abortSignal|await|break))/, reason: 'Unbounded while(true) loop without await/break/abortSignal detected.' },
];

/**
 * Validates and compiles an LLM-synthesized JavaScript async function body
 * inside an isolated `node:vm` sandbox context.
 *
 * Expected signature of synthesized code body:
 * Assumes `(args, ctx)` are in scope, and returns `{ success: boolean, summary: string, data?: any }`.
 */
export function validateAndCompileSynthesizedTool(
  toolName: string,
  functionBodyCode: string
): SandboxValidationResult {
  if (!/^[a-z][a-z0-9_]{2,45}$/.test(toolName)) {
    return {
      valid: false,
      error: `Invalid tool name "${toolName}". Must be lowercase snake_case (3-46 chars).`,
    };
  }

  for (const rule of FORBIDDEN_PATTERNS) {
    if (rule.pattern.test(functionBodyCode)) {
      return {
        valid: false,
        error: rule.reason,
      };
    }
  }

  const wrappedSource = `
    (async function synthesized_${toolName}(args, ctx) {
      "use strict";
      ${functionBodyCode}
    })
  `;

  try {
    // 1. Syntax & compilation check via node:vm
    const script = new vm.Script(wrappedSource, {
      filename: `synthesized_${toolName}.vm.js`,
    });

    const sandboxGlobals = {
      Vec3,
      Math,
      Date,
      JSON,
      Promise,
      setTimeout,
      clearTimeout,
      console: {
        log: (...m: any[]) => console.log(`[Skill:${toolName}]`, ...m),
        warn: (...m: any[]) => console.warn(`[Skill:${toolName}]`, ...m),
      },
    };

    const vmContext = vm.createContext(sandboxGlobals);
    const compiledFn = script.runInContext(vmContext, { timeout: 1000 });

    if (typeof compiledFn !== 'function') {
      return {
        valid: false,
        error: 'Synthesized script did not evaluate to a callable function.',
      };
    }

    const safeHandler: MCPToolHandler = async (
      args: Record<string, any>,
      ctx: MCPToolExecutionContext
    ) => {
      const result = await compiledFn(args || {}, ctx);
      if (result && typeof result.success === 'boolean' && typeof result.summary === 'string') {
        return result;
      }
      return {
        success: true,
        summary: typeof result === 'string' ? result : `Completed synthesized tool ${toolName}.`,
        data: result,
      };
    };

    return {
      valid: true,
      compiledHandler: safeHandler,
    };
  } catch (err: any) {
    return {
      valid: false,
      error: `Syntax/VM Compilation Error: ${err.message}`,
    };
  }
}
