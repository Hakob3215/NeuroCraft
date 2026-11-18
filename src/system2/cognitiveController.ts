import { CognitiveDecision, ReflexInterruptEvent, SubtaskStatus } from '../types';
import { System1BotController } from '../system1/botController';
import { DynamicMCPToolRegistry } from './mcp/toolRegistry';
import { NeuroCraftMCPServer } from './mcp/mcpServer';
import { HighLevelMetaAgent } from './meta/metaAgent';
import { SelfSynthesizingToolFactory } from './meta/toolFactory';
import { ExecutedStepRecord, UnifiedLLMClient } from './llmClient';
import { EpisodicMemoryStore } from './memory/memoryStore';
import { HumanizedChatEngine } from './chatEngine';
import { HumanIdleSimulator } from './idleFsm';
import { SpeculativePrefetcher } from './speculative';
import { registerBaseMCPTools } from './skills/baseSkills';
import { logVerbose } from '../logger';

/**
 * System 2: PIANO Cognitive Controller & Autonomous Multi-Step Goal Engine.
 * Coordinates perception snapshots, the High-Level Meta-Agent, the Dynamic MCP Server,
 * Speculative Prefetching, and continuous ReAct step chaining until long-term goals finish.
 */
export class CognitiveController {
  public readonly registry: DynamicMCPToolRegistry;
  public readonly mcpServer: NeuroCraftMCPServer;
  public readonly memory: EpisodicMemoryStore;
  public readonly llm: UnifiedLLMClient;
  public readonly toolFactory: SelfSynthesizingToolFactory;
  public readonly metaAgent: HighLevelMetaAgent;
  public readonly chatEngine: HumanizedChatEngine;
  public readonly idleSimulator: HumanIdleSimulator;
  public readonly prefetcher: SpeculativePrefetcher;

  private activeAbortController: AbortController | null = null;
  private activeGoal: string | null = null;
  private stepHistory: ExecutedStepRecord[] = [];
  private goalLoopId = 0;

  constructor(private readonly system1: System1BotController) {
    this.registry = new DynamicMCPToolRegistry();
    registerBaseMCPTools(this.registry);

    this.mcpServer = new NeuroCraftMCPServer(this.registry);
    this.memory = new EpisodicMemoryStore();
    this.llm = new UnifiedLLMClient();
    this.toolFactory = new SelfSynthesizingToolFactory(this.registry, this.llm);
    this.metaAgent = new HighLevelMetaAgent(
      this.registry,
      this.toolFactory,
      this.llm,
      this.memory
    );
    this.chatEngine = new HumanizedChatEngine(() => this.system1.bot);
    this.idleSimulator = new HumanIdleSimulator(this.system1, (toolName, args) =>
      this.executeMCPToolCall(toolName, args)
    );
    this.idleSimulator.startAutonomousIdleLoop();

    this.prefetcher = new SpeculativePrefetcher((subtask) => {
      console.log(
        `[SpeculativePrefetch] Subtask "${subtask.toolName}" reached ${(subtask.progress * 100).toFixed(0)}% completion. Ready for zero-latency transition.`
      );
    });

    this.mcpServer.onNotification((notif) => {
      console.log(
        `[MCP Bus] Notification emitted: ${notif.method} -> Tool="${notif.params.toolName}" (Total tools: ${notif.params.totalTools})`
      );
    });

    this.system1.onDisconnect((reason: string) => {
      console.warn(`[CognitiveController] Agent disconnected from Minecraft server (${reason}). Aborting active goal.`);
      this.cancelActiveGoal();
      if (this.activeAbortController) {
        this.activeAbortController.abort();
      }
      this.idleSimulator.stopAutonomousIdleLoop();
    });
  }

  /**
   * Handles incoming player chat and initiates the Autonomous Multi-Step Goal Loop
   * whenever a multi-step or long-term objective is detected.
   */
  public async handlePlayerChat(speaker: string, message: string): Promise<void> {
    console.log(`[Chat <- World] <${speaker}> ${message}`);
    const lower = message.toLowerCase().trim();

    // Immediate cancellation check
    if (
      lower === 'stop' ||
      lower === 'cancel' ||
      lower === 'halt' ||
      lower === 'quit' ||
      lower.includes('quit that') ||
      lower.includes('stop that') ||
      lower.includes('cancel that')
    ) {
      this.cancelActiveGoal();
      await this.executeMCPToolCall('stop_current_action', {});
      await this.chatEngine.sayHumanized(`Stopped and cleared my active goal, ${speaker}!`);
      return;
    }

    // Preempt any running background GoalLoop immediately so it never races with the new player command
    if (this.activeGoal) {
      this.cancelActiveGoal();
    }

    const state = this.system1.getLatestWorldState();
    logVerbose('WORLD_STATE', `World state snapshot on chat from <${speaker}>`, state);
    this.idleSimulator.startThinkingBehavior(speaker);

    try {
      const decision = await this.metaAgent.evaluatePlayerCommand(
        speaker,
        message,
        state,
        () => {
          this.chatEngine
            .sayHumanized(`Give me a second ${speaker}, writing a custom skill for that...`)
            .catch(() => {});
        }
      );

      this.idleSimulator.stopThinkingBehavior();
      logVerbose('DECISION', `Initial cognitive decision for <${speaker}>`, decision);
      console.log(`[CognitiveController] Thought: ${decision.thought}`);

      // Sanitize bogus activeGoal strings like "None", "null", "N/A"
      if (
        decision.activeGoal &&
        /^(none|null|n\/a|undefined|nothing|chat|greeting)$/i.test(decision.activeGoal.trim())
      ) {
        decision.activeGoal = undefined;
        if (!decision.toolCall) {
          decision.goalCompleted = true;
        }
      }

      // Sanitize invalid toolCall objects where name is missing or "undefined"/"null"/"none"
      if (
        decision.toolCall &&
        (!decision.toolCall.name ||
          /^(undefined|null|none|n\/a)$/i.test(String(decision.toolCall.name).trim()))
      ) {
        decision.toolCall = undefined;
      }

      const terminalSingleTools = new Set([
        'follow_player',
        'come_here',
        'stop_current_action',
        'give_item_to_player',
        'inspect_surroundings',
        'scan_area',
        'neural_chunk_scan',
        'collect_dropped_items',
        'tame_animal',
        'pillar_up',
        'build_simple_house',
      ]);

      const isMultiStageMessage =
        /\b(end goal|long term|full iron|iron armor|diamond|smelt|and then|after that|everything you|intermediate steps)\b/i.test(
          message
        );

      const isSelfContainedSingleStep =
        !isMultiStageMessage &&
        (!Array.isArray(decision.planSteps) || decision.planSteps.length <= 1) &&
        decision.toolCall &&
        (terminalSingleTools.has(decision.toolCall.name) ||
          decision.toolCall.name === 'engage_combat' ||
          Boolean(this.registry.getTool(decision.toolCall.name)?.schema.isSynthesized));

      // Determine whether this command initiates a multi-step autonomous goal
      const isMultiStepIntent =
        !isSelfContainedSingleStep &&
        (Boolean(decision.activeGoal) ||
          (decision.goalCompleted === false && Boolean(decision.toolCall)) ||
          (Array.isArray(decision.planSteps) && decision.planSteps.length > 1) ||
          /\b(step 1|first step|then |first,)\b/i.test(decision.thought) ||
          isMultiStageMessage);

      // Start speech asynchronously so it does NOT block or delay motor execution
      if (decision.speechResponse) {
        this.chatEngine.sayHumanized(decision.speechResponse).catch(() => {});
      }

      if (isMultiStepIntent && decision.toolCall?.name !== 'stop_current_action') {
        const goalDescription = decision.activeGoal || message;
        this.startNewAutonomousGoal(goalDescription, decision);
      } else {
        this.cancelActiveGoal();
        if (decision.toolCall) {
          await this.executeMCPToolCall(decision.toolCall.name, decision.toolCall.arguments);
        }
      }
    } catch (err: any) {
      this.idleSimulator.stopThinkingBehavior();
      console.warn(`[CognitiveController] Error handling chat: ${err.message}`);
    }
  }

  private cancelActiveGoal(): void {
    this.activeGoal = null;
    this.goalLoopId++;
    this.idleSimulator.setGoalActive(false);
    if (this.activeAbortController) {
      this.activeAbortController.abort();
      this.activeAbortController = null;
    }
  }

  /**
   * Runs a multi-step goal autonomously from start to finish (up to 45 sequential subtasks)
   * until `goalCompleted === true` or the player cancels it.
   */
  private startNewAutonomousGoal(
    goalDescription: string,
    initialDecision: CognitiveDecision
  ): void {
    this.cancelActiveGoal();
    this.activeGoal = goalDescription;
    this.stepHistory = [];
    const currentLoopId = ++this.goalLoopId;
    this.idleSimulator.setGoalActive(true);

    console.log(`[GoalLoop] Started autonomous goal: "${goalDescription}"`);
    if (initialDecision.planSteps && initialDecision.planSteps.length > 0) {
      console.log(`[GoalLoop] Planned Roadmap: ${initialDecision.planSteps.join(' -> ')}`);
    }

    (async () => {
      // Execute Step 1 from the initial decision if present
      if (initialDecision.toolCall) {
        const step1Result = await this.executeMCPToolCall(
          initialDecision.toolCall.name,
          initialDecision.toolCall.arguments
        );
        this.stepHistory.push({
          stepNumber: 1,
          toolName: initialDecision.toolCall.name,
          args: initialDecision.toolCall.arguments,
          success: step1Result.success,
          summary: step1Result.summary,
        });
      }

      const maxSteps = 45;
      for (let stepNum = this.stepHistory.length + 1; stepNum <= maxSteps; stepNum++) {
        if (this.goalLoopId !== currentLoopId || !this.activeGoal) {
          break;
        }

        // Brief 50ms breathing tick so inventory packet updates settle cleanly
        await new Promise((r) => setTimeout(r, 50));
        if (this.goalLoopId !== currentLoopId || !this.activeGoal) break;

        const freshState = this.system1.getLatestWorldState();
        logVerbose('WORLD_STATE', `World state snapshot before GoalLoop Step #${stepNum}`, freshState);
        console.log(
          `\n[GoalLoop Step #${stepNum}] Evaluating next step for goal: "${this.activeGoal}"...`
        );

        const nextDecision = await this.metaAgent.evaluateGoalContinuation(
          this.activeGoal,
          this.stepHistory,
          freshState
        );

        if (this.goalLoopId !== currentLoopId || !this.activeGoal) break;

        logVerbose('DECISION', `GoalLoop Step #${stepNum} decision`, nextDecision);
        console.log(`[GoalLoop Step #${stepNum}] Thought: ${nextDecision.thought}`);

        if (nextDecision.speechResponse) {
          this.chatEngine.sayHumanized(nextDecision.speechResponse).catch(() => {});
        }

        if (
          nextDecision.toolCall &&
          nextDecision.toolCall.name &&
          !/^(undefined|null|none)$/i.test(String(nextDecision.toolCall.name).trim())
        ) {
          const execRes = await this.executeMCPToolCall(
            nextDecision.toolCall.name,
            nextDecision.toolCall.arguments || {}
          );
          this.stepHistory.push({
            stepNumber: stepNum,
            toolName: nextDecision.toolCall.name,
            args: nextDecision.toolCall.arguments || {},
            success: execRes.success,
            summary: execRes.summary,
          });
        } else if (!nextDecision.goalCompleted) {
          console.log(
            `[GoalLoop] Step #${stepNum} returned no valid toolCall; concluding goal "${this.activeGoal}".`
          );
          this.idleSimulator.setGoalActive(false);
          this.activeGoal = null;
          break;
        }

        if (nextDecision.goalCompleted) {
          console.log(`[GoalLoop] Goal "${this.activeGoal}" COMPLETED after ${stepNum} steps!`);
          this.idleSimulator.setGoalActive(false);
          if (!nextDecision.speechResponse) {
            this.chatEngine
              .sayHumanized(`Finished my long-term goal: ${this.activeGoal}!`)
              .catch(() => {});
          }
          this.activeGoal = null;
          break;
        }
      }
    })().catch((err) => {
      console.warn(`[GoalLoop] Autonomous loop error: ${err.message}`);
    });
  }

  /**
   * Executes a tool through the JSON-RPC 2.0 MCP Server and returns `{ success, summary }`
   * so the Autonomous Goal Loop can inspect the outcome and adapt on the next step.
   */
  public async executeMCPToolCall(
    toolName: string,
    args: Record<string, any>
  ): Promise<{ success: boolean; summary: string }> {
    if (this.activeAbortController) {
      this.activeAbortController.abort();
    }
    this.activeAbortController = new AbortController();
    const signal = this.activeAbortController.signal;

    const subtask: SubtaskStatus = {
      id: `task_${Date.now()}`,
      toolName,
      description: `Executing MCP tool ${toolName}`,
      args,
      startedAt: Date.now(),
      estimatedDurationMs: 4000,
      progress: 0,
      status: 'RUNNING',
    };

    this.system1.setSubtaskStatus(subtask);
    console.log(`[MCP Call] -> ${toolName}(${JSON.stringify(args)})`);

    const rpcResponse = await this.mcpServer.handleRpcRequest(
      {
        jsonrpc: '2.0',
        id: subtask.id,
        method: 'tools/call',
        params: {
          name: toolName,
          arguments: args,
        },
      },
      () =>
        this.system1.createToolExecutionContext(
          subtask,
          (msg) => this.chatEngine.sayHumanized(msg),
          signal
        )
    );

    this.system1.setSubtaskStatus(null);

    if (rpcResponse.error) {
      subtask.status = 'FAILED';
      subtask.error = rpcResponse.error.message;
      console.warn(`[MCP Execution] Tool "${toolName}" error: ${rpcResponse.error.message}`);
      return { success: false, summary: rpcResponse.error.message };
    }

    subtask.progress = 1.0;
    subtask.status = 'COMPLETED';
    const textPayload = rpcResponse.result?.content?.[0]?.text;
    if (textPayload) {
      try {
        const parsed = JSON.parse(textPayload);
        console.log(`[MCP Execution] ${parsed.summary}`);
        this.memory.addMemory(
          'GOAL_COMPLETED',
          parsed.summary,
          [toolName],
          this.system1.getLatestWorldState().position
        );
        return {
          success: Boolean(parsed.success),
          summary: String(parsed.summary || `Completed ${toolName}`),
        };
      } catch {
        return { success: true, summary: textPayload };
      }
    }

    return { success: true, summary: `Completed ${toolName}` };
  }

  public handleReflexInterrupt(event: ReflexInterruptEvent): void {
    console.log(
      `[ReflexInterrupt -> System 2] [${event.severity}] ${event.hazard}: ${event.details}`
    );

    if (event.severity === 'CRITICAL' && this.activeAbortController) {
      this.activeAbortController.abort();
      this.activeAbortController = null;
    }

    this.memory.addMemory(
      'LANDMARK',
      `Encountered hazard ${event.hazard}: ${event.details}`,
      ['hazard', event.hazard.toLowerCase()],
      event.position
    );
  }

  public handleSubtaskProgress(subtask: SubtaskStatus): void {
    this.prefetcher.onSubtaskUpdate(subtask);
  }
}
