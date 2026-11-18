import { CognitiveDecision, SanitizedWorldState } from '../../types';
import { DynamicMCPToolRegistry } from '../mcp/toolRegistry';
import { SelfSynthesizingToolFactory } from './toolFactory';
import { ExecutedStepRecord, UnifiedLLMClient } from '../llmClient';
import { EpisodicMemoryStore } from '../memory/memoryStore';

/**
 * High-Level Meta-Agent Overseer.
 * Monitors all incoming commands and chat messages across the Minecraft world,
 * manages multi-step autonomous goal continuations, and invokes the
 * Self-Synthesizing Tool Factory whenever a novel capability is needed.
 */
export class HighLevelMetaAgent {
  constructor(
    private readonly registry: DynamicMCPToolRegistry,
    private readonly toolFactory: SelfSynthesizingToolFactory,
    private readonly llm: UnifiedLLMClient,
    private readonly memory: EpisodicMemoryStore
  ) {}

  public async evaluatePlayerCommand(
    speaker: string,
    message: string,
    worldState: SanitizedWorldState,
    onSynthesisStart?: () => void
  ): Promise<CognitiveDecision> {
    this.memory.recordChat(speaker, message);
    const trimmed = message.trim();
    const lower = trimmed.toLowerCase();

    if (lower === '!tools' || lower === 'agent tools' || lower === 'list tools') {
      const all = this.registry.listSchemas();
      const synth = this.registry.listSynthesizedSchemas();
      return {
        thought: 'Listing registered MCP tools.',
        speechResponse: `MCP Catalog: ${all.length} tools active (${synth.length} self-synthesized: ${
          synth.map((s) => s.name).join(', ') || 'none yet'
        }).`,
        goalCompleted: true,
      };
    }

    // If the user explicitly asks to remember current coordinates, store landmark immediately
    if (lower.includes('remember') && (lower.includes('coord') || lower.includes('here') || lower.includes('location') || lower.includes('spot'))) {
      this.memory.addMemory(
        'LANDMARK',
        `Saved landmark requested by ${speaker} at (${worldState.position.x}, ${worldState.position.y}, ${worldState.position.z})`,
        ['landmark', 'coords', 'coordinates', 'home', 'base', 'start', speaker],
        worldState.position
      );
    }

    const relevantMemories = this.memory
      .queryRelevantMemories(trimmed, 6)
      .map((m) => m.summary);

    const decision = await this.llm.generateCognitiveDecision(
      speaker,
      trimmed,
      worldState,
      this.registry.listSchemas(),
      relevantMemories
    );

    if (decision.rememberLandmark?.position) {
      this.memory.addMemory(
        'LANDMARK',
        `Landmark "${decision.rememberLandmark.name}" at (${decision.rememberLandmark.position.x}, ${decision.rememberLandmark.position.y}, ${decision.rememberLandmark.position.z})`,
        ['landmark', 'coords', decision.rememberLandmark.name],
        decision.rememberLandmark.position
      );
    }

    return this.resolveSynthesisIfNeeded(decision, speaker, onSynthesisStart);
  }

  /**
   * Evaluates the next step in an ongoing autonomous multi-step goal loop.
   */
  public async evaluateGoalContinuation(
    activeGoal: string,
    stepHistory: ExecutedStepRecord[],
    worldState: SanitizedWorldState,
    onSynthesisStart?: () => void
  ): Promise<CognitiveDecision> {
    const relevantMemories = this.memory
      .queryRelevantMemories(activeGoal, 6)
      .map((m) => m.summary);

    const decision = await this.llm.generateGoalContinuationDecision(
      activeGoal,
      stepHistory,
      worldState,
      this.registry.listSchemas(),
      relevantMemories
    );

    return this.resolveSynthesisIfNeeded(decision, 'GoalLoop', onSynthesisStart);
  }

  private async resolveSynthesisIfNeeded(
    decision: CognitiveDecision,
    speaker: string,
    onSynthesisStart?: () => void
  ): Promise<CognitiveDecision> {
    // Auto-convert unregistered toolCall names into a synthesizeRequest so novel tools are synthesized automatically
    if (
      !decision.synthesizeRequest &&
      decision.toolCall &&
      decision.toolCall.name &&
      !/^(undefined|null|none|n\/a)$/i.test(String(decision.toolCall.name).trim()) &&
      !this.registry.getTool(decision.toolCall.name)
    ) {
      const rawName = String(decision.toolCall.name)
        .toLowerCase()
        .replace(/[^a-z0-9_]/g, '_');
      decision.synthesizeRequest = {
        desiredToolName: rawName,
        taskDescription: decision.activeGoal || decision.thought || `Execute ${rawName}`,
        requiredParameters: Object.fromEntries(
          Object.keys(decision.toolCall.arguments || {}).map((k) => [k, 'any'])
        ),
        triggeringAsk: decision.activeGoal || decision.thought || rawName,
      };
    }

    if (!decision.synthesizeRequest) {
      return decision;
    }

    if (onSynthesisStart) onSynthesisStart();

    const req = decision.synthesizeRequest;
    const synthResult = await this.toolFactory.synthesizeAndRegisterTool(
      req.desiredToolName,
      req.taskDescription,
      req.requiredParameters,
      req.triggeringAsk
    );

    if (synthResult.success) {
      this.memory.addMemory(
        'SYNTHESIZED_SKILL',
        `Synthesized new MCP tool "${synthResult.toolName}" for request: "${req.triggeringAsk}"`,
        ['synthesized', synthResult.toolName, speaker]
      );

      // Preserve any arguments from the original toolCall or non-type literal values in requiredParameters
      const extractedParams: Record<string, any> = {};
      if (req.requiredParameters && typeof req.requiredParameters === 'object') {
        for (const [k, v] of Object.entries(req.requiredParameters)) {
          if (
            typeof v !== 'string' ||
            !/^(string|number|boolean|object|any|array)$/i.test(v.trim())
          ) {
            extractedParams[k] = v;
          }
        }
      }
      const mergedArgs = {
        ...extractedParams,
        ...(decision.toolCall?.arguments || {}),
        speaker,
      };

      return {
        ...decision,
        thought: `${decision.thought} -> Synthesized and hot-reloaded MCP tool "${synthResult.toolName}".`,
        speechResponse:
          decision.speechResponse ||
          `I just synthesized a new MCP skill (${synthResult.toolName}) for this step!`,
        toolCall: {
          name: synthResult.toolName,
          arguments: mergedArgs,
        },
      };
    }

    return {
      ...decision,
      thought: `Attempted to synthesize "${req.desiredToolName}", but sandbox rejected: ${synthResult.error}`,
      speechResponse: `My safety sandbox caught an issue while drafting "${req.desiredToolName}" (${synthResult.error}). Adapting plan...`,
    };
  }
}
