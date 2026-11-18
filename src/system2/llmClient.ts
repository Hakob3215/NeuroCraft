import { CONFIG } from '../config';
import { CognitiveDecision, MCPToolSchema, SanitizedWorldState } from '../types';
import { logVerbose } from '../logger';
import { sharedNeuralBridge } from '../system1/neural/neuralBridge';
import { DynamicMCPToolRegistry } from './mcp/toolRegistry';
import { computeGroundedAffordances } from './goalPlanner';

export interface ExecutedStepRecord {
  stepNumber: number;
  toolName: string;
  args: Record<string, any>;
  success: boolean;
  summary: string;
}

/**
 * Unified LLM Client supporting:
 * 1. Google Gemini API (GEMINI_API_KEY - defaults to gemini-3.8-flash)
 * 2. OpenAI Compatible API (OPENAI_API_KEY)
 * 3. Local Ollama SLM (http://127.0.0.1:11434)
 * 4. Built-in Autonomous Tech-Tree & Heuristic Fallback
 */
export class UnifiedLLMClient {
  private cachedGeminiModelId: string | null = null;
  private thinkingConfigUnsupported = new Set<string>();

  constructor(private readonly options?: { offlineOnly?: boolean }) {}

  public async generateCognitiveDecision(
    speaker: string,
    userMessage: string,
    worldState: SanitizedWorldState,
    availableTools: MCPToolSchema[],
    recentMemories: string[]
  ): Promise<CognitiveDecision> {
    // 0. Ultra-Fast Symbolic/Reflex Intent Router (<1ms) for unambiguous single-step commands
    if (process.env.DISABLE_FAST_INTENT !== 'true') {
      const fastDecision = this.tryFastIntentDecision(speaker, userMessage, worldState);
      if (fastDecision) {
        console.log(`[Fast-Path Reflex] Resolved "${userMessage}" in <1ms -> ${fastDecision.toolCall?.name}`);
        logVerbose('FAST_INTENT', `Resolved unambiguous command in <1ms without LLM wait`, fastDecision);
        return fastDecision;
      }
    }

    const systemPrompt = this.buildDecisionSystemPrompt(availableTools, worldState, recentMemories);
    const compactSystemPrompt = this.buildCompactLocalSystemPrompt(availableTools, worldState, recentMemories);
    const userPrompt = `Player "${speaker}" says: "${userMessage}"
If this request requires multiple sequential steps (such as getting full iron armor, building a base, or progressing through the tech tree):
- Set "activeGoal" to a clear description of the end goal (including any coordinates to remember).
- Set "goalCompleted": false so the autonomous execution loop continues after this first toolCall finishes!
- Set "planSteps" to the ordered list of steps.
- Set "toolCall" to Step 1 right now.
Respond strictly with a valid JSON object matching the CognitiveDecision schema.`;

    const compactUserPrompt = `Player "${speaker}" says: "${userMessage}"
Return ONLY a valid JSON object: {"thought":"...", "toolCall":{"name":"...", "arguments":{}}}`;

    const rawJson = await this.callConfiguredModel(
      systemPrompt,
      userPrompt,
      compactSystemPrompt,
      compactUserPrompt,
      140,
      360,
      worldState.botUsername || 'default'
    );
    if (rawJson) {
      const parsed = this.tryParseJson<CognitiveDecision>(rawJson);
      if (parsed && (parsed.thought || parsed.toolCall)) {
        if (!parsed.thought) parsed.thought = `Executing ${parsed.toolCall?.name || 'response'}.`;
        return parsed;
      }
    }

    if (process.env.STRICT_LLM_BENCHMARK === 'true') {
      return {
        thought: `Strict LLM benchmark mode: model failed to return valid JSON (${rawJson ? rawJson.slice(0, 80) : 'null'}).`,
        goalCompleted: true,
      };
    }

    return this.fallbackHeuristicDecision(speaker, userMessage, worldState, availableTools);
  }

  /**
   * Autonomous ReAct Continuation Planner:
   * Called automatically after each subtask finishes (or fails) while `activeGoal` is in progress.
   */
  public async generateGoalContinuationDecision(
    activeGoal: string,
    stepHistory: ExecutedStepRecord[],
    worldState: SanitizedWorldState,
    availableTools: MCPToolSchema[],
    recentMemories: string[]
  ): Promise<CognitiveDecision> {
    const systemPrompt = this.buildDecisionSystemPrompt(availableTools, worldState, recentMemories);

    const historyText = stepHistory
      .slice(-10)
      .map(
        (s) =>
          `Step ${s.stepNumber}: ${s.toolName}(${JSON.stringify(s.args)}) -> ${
            s.success ? 'SUCCESS' : 'FAILED'
          }: ${s.summary}`
      )
      .join('\n');

    const userPrompt = `AUTONOMOUS GOAL LOOP CONTINUATION
Active End Goal: "${activeGoal}"

Recent Executed Steps:
${historyText || '(None yet)'}

Minecraft Tech-Tree Reference:
- 1 *_log -> 4 *_planks; 2 *_planks -> 4 stick; 4 *_planks -> 1 crafting_table.
- wooden_pickaxe requires 3 *_planks + 2 stick (requires crafting_table).
- Mine stone with wooden_pickaxe -> cobblestone.
- stone_pickaxe requires 3 cobblestone + 2 stick.
- furnace requires 8 cobblestone.
- Mine iron_ore with stone_pickaxe (NEVER wooden_pickaxe!) -> raw_iron; mine coal_ore -> coal.
- Mine diamond_ore with iron_pickaxe (or diamond/netherite pickaxe; NEVER stone or wooden pickaxe!) -> diamond.
- Smelt raw_iron in furnace using coal/planks/logs (` + '`smelt_item`' + `) -> iron_ingot.
- Full iron armor requires 24 iron_ingot total: iron_helmet (5), iron_chestplate (8), iron_leggings (7), iron_boots (4).
- Equip armor using ` + '`equip_armor_or_item`' + ` with itemName="all_armor".

Instructions:
1. Check current Inventory and Recent Executed Steps carefully. Compare what the previous step reported against the actual Inventory!
2. TOOL DURABILITY & PRESERVATION:
   - Check tool durability numbers! High-tier pickaxes (iron_pickaxe, diamond_pickaxe) MUST be saved for rare ores (diamond_ore, gold_ore).
   - NEVER waste an iron_pickaxe digging ordinary stone or tunneling down—craft and use stone_pickaxe(s) for general excavation!
   - Before digging deep staircase tunnels toward diamond level (Y=-58), ensure you have multiple stone_pickaxes or cobblestone + sticks to craft them.
3. Verify Tool Availability: Before calling gather_resource for iron_ore, verify you have a stone_pickaxe (or iron_pickaxe) in Inventory, OR keep spare logs/planks/sticks in Inventory so gather_resource can auto-craft a replacement pickaxe if yours breaks mid-mine!
3. Batching: When gathering underground iron_ore, prefer batches of count: 8 to 12 per step (the 3D vein miner will automatically clear full connected veins).
4. If the Active End Goal is 100% accomplished, set "goalCompleted": true and provide a celebratory "speechResponse".
5. Otherwise, set "goalCompleted": false and output the immediate next "toolCall" (or "synthesizeRequest").
6. SELF-ADAPTATION: If a tool failed or did not yield the expected inventory change, adapt immediately (e.g. craft a replacement pickaxe if yours broke, call "explore_or_dig_for_ores" with mode="staircase_mine", or use "synthesizeRequest" to write/update a custom MCP tool tailored to the current terrain/situation!).
7. Only set "speechResponse" if announcing a major milestone or completion (leave "speechResponse" undefined on routine intermediate steps so chat isn't spammed).
Respond strictly with valid JSON.`;

    const compactSystemPrompt = this.buildCompactLocalSystemPrompt(
      availableTools,
      worldState,
      recentMemories,
      stepHistory
    );
    const compactUserPrompt = `Active Goal: "${activeGoal}"
Recent Steps:
${historyText || '(None)'}
Return ONLY a valid JSON object: {"thought":"...", "toolCall":{"name":"...", "arguments":{}}}`;

    const rawJson = await this.callConfiguredModel(
      systemPrompt,
      userPrompt,
      compactSystemPrompt,
      compactUserPrompt,
      140,
      360,
      worldState.botUsername || 'default'
    );
    if (rawJson) {
      const parsed = this.tryParseJson<CognitiveDecision>(rawJson);
      if (parsed && (parsed.thought || parsed.toolCall || parsed.goalCompleted !== undefined)) {
        if (!parsed.thought) parsed.thought = 'Continuing goal execution.';
        return parsed;
      }
    }

    if (process.env.STRICT_LLM_BENCHMARK === 'true') {
      return {
        thought: `Strict LLM benchmark mode: model failed to return valid continuation JSON.`,
        goalCompleted: true,
      };
    }

    return this.fallbackGoalContinuation(activeGoal, stepHistory, worldState);
  }

  public async synthesizeToolCode(
    desiredToolName: string,
    taskDescription: string,
    requiredParameters: Record<string, string>,
    triggeringAsk: string
  ): Promise<{
    description: string;
    parameters: Record<string, { type: string; description: string; default?: any }>;
    functionBodyCode: string;
    tags: string[];
  }> {
    const sysPrompt = `You are the NeuroCraft Self-Synthesizing MCP Tool Factory.
Write the body of an async JavaScript function with signature (args, ctx) => Promise<{ success: boolean, summary: string }>.
Available in scope:
- args: object containing tool arguments
- ctx.bot: Mineflayer bot instance (ctx.bot.entity.position, ctx.bot.blockAt, ctx.bot.findBlocks, ctx.bot.findBlock, ctx.bot.placeBlock, ctx.bot.craft, ctx.bot.openFurnace, ctx.bot.setControlState, etc.)
- ctx.kinematics.pathfindTo(targetVec3, stopDist = 2.2): A* pathfinding helper that walks to within stopDist blocks of targetVec3 and returns Promise<boolean>. ALWAYS use this to navigate to blocks or coordinates instead of moveAlongSpline!
- ctx.kinematics.safeDig(blockOrVec3): Stops movement, settles on ground, equips the right tool, walks closer if needed, and safely mines the block with a watchdog timeout, returning Promise<boolean>. ALWAYS use ctx.kinematics.safeDig(blockOrVec3) instead of raw ctx.bot.dig()!
- ctx.kinematics.safePlaceBlock(targetVec3, itemName, allowScaffold = true): High-reliability, reach-aware block placement helper that automatically navigates within 1.5 - 3.2m of targetVec3, prevents body collision, finds reference blocks (or scaffolds below if floating), and places reliably. ALWAYS use this instead of raw ctx.bot.placeBlock! NEVER use a single static safeStand observation coordinate!
- ctx.kinematics.lookAtSmooth(vec3, durationMs)
- ctx.kinematics.moveAlongSpline(waypointsVec3Array, sprintBoolean)
- ctx.kinematics.strikeTarget(entity)
- ctx.kinematics.equipItemHumanized(itemName, 'hand'|'head'|'torso'|'legs'|'feet'|'off-hand')
- ctx.world.getGroundHeight(x, z, startY): Scans downwards to find the topmost solid ground block at (x, z). ALWAYS use this to anchor structures directly to terrain height instead of guessing bot.entity.position.y!
- ctx.world.scanAreaTopology(minX, minZ, maxX, maxZ): Area topology scan returning minHeight, maxHeight, medianHeight, isFlat, and obstruction counts.
- ctx.world.getTopSolidBlock(x, z, startY): Returns top solid block instance at (x, z).
- ctx.worldState(): returns SanitizedWorldState
- ctx.reportProgress(0.0 to 1.0)
- ctx.sayHumanized(text)
- ctx.abortSignal: check if (ctx.abortSignal.aborted) break;
- Vec3(x, y, z)
CRITICAL MINEFLAYER RULE: When calling ctx.bot.findBlock({ matching }) or ctx.bot.findBlocks({ matching }), the matching callback MUST ONLY check b?.name (NEVER check b.position inside matching, because Mineflayer's chunk palette check passes a prototype block with b.position === null!).
Do NOT use require(), process.exit, or child_process.
Return ONLY valid JSON:
{
  "description": "...",
  "parameters": { "paramName": { "type": "number|string|boolean", "description": "...", "default": ... } },
  "functionBodyCode": "...",
  "tags": ["tag1", "tag2"]
}`;

    const usrPrompt = `Synthesize new MCP tool "${desiredToolName}".
Task description: ${taskDescription}
Original user request: "${triggeringAsk}"
Suggested parameters: ${JSON.stringify(requiredParameters)}`;

    const rawJson = await this.callConfiguredModel(sysPrompt, usrPrompt, undefined, undefined, 512, 2048);
    if (rawJson) {
      const parsed = this.tryParseJson<any>(rawJson);
      if (parsed && typeof parsed.functionBodyCode === 'string') {
        return {
          description: parsed.description || taskDescription,
          parameters: parsed.parameters || {},
          functionBodyCode: parsed.functionBodyCode,
          tags: Array.isArray(parsed.tags) ? parsed.tags : [desiredToolName],
        };
      }
    }

    return this.fallbackSynthesizeTool(desiredToolName, taskDescription, triggeringAsk);
  }

  private async callConfiguredModel(
    systemPrompt: string,
    userPrompt: string,
    compactSystemPrompt?: string,
    compactUserPrompt?: string,
    localMaxTokens = 160,
    geminiMaxTokens = 360,
    botId = 'default'
  ): Promise<string | null> {
    if (this.options?.offlineOnly || process.env.LLM_OFFLINE_ONLY === 'true') {
      return null;
    }
    const apiKey = CONFIG.llm.apiKey;
    const provider = (process.env.LLM_PROVIDER as any) || CONFIG.llm.provider;
    logVerbose('LLM_PROMPT', 'Sending prompt to configured LLM', {
      provider,
      configuredModel: CONFIG.llm.model,
      systemPrompt: compactSystemPrompt || systemPrompt,
      userPrompt: compactUserPrompt || userPrompt,
    });

    // 0. Only use Local GPU 1.5B SLM first if NEURAL_FORCE_LOCAL_SLM=true is explicitly set.
    // Otherwise, prefer Gemini 3.8 Flash (~0.8-1.5s, frontier intelligence) for System 2 while RTX 5060 powers System 1 (2ms reflexes & 5ms 3D chunk scans),
    // keeping the Local 1.5B SLM as an automatic offline fallback at step 4!
    if (process.env.NEURAL_FORCE_LOCAL_SLM === 'true') {
      const gpuOut = await sharedNeuralBridge.generateLocalSLMResponse(
        compactSystemPrompt || systemPrompt,
        compactUserPrompt || userPrompt,
        localMaxTokens,
        undefined,
        undefined,
        undefined,
        undefined,
        botId
      );
      if (gpuOut && gpuOut.content) {
        return gpuOut.content;
      }
      if (process.env.STRICT_LLM_BENCHMARK === 'true') {
        return null;
      }
    }

    if (process.env.STRICT_LLM_BENCHMARK === 'true' && process.env.LLM_PROVIDER === 'gemini') {
      // Only test Gemini in strict Gemini mode (do not fall through to local SLM if Gemini fails)
    }

    // 1. Try Gemini if API key is provided
    const geminiKey = process.env.GEMINI_API_KEY?.trim();
    if (
      (provider === 'gemini' || provider === 'auto' || provider === 'local_gpu') &&
      geminiKey &&
      geminiKey !== 'PASTE_YOUR_GEMINI_API_KEY_HERE'
    ) {
      const normalizedConfigured = CONFIG.llm.model
        .trim()
        .toLowerCase()
        .replace(/\s+/g, '-')
        .replace(/^(?!gemini-)/, 'gemini-');

      const candidateModels = [
        ...(this.cachedGeminiModelId ? [this.cachedGeminiModelId] : []),
        normalizedConfigured,
        CONFIG.llm.model.trim(),
        'gemini-3.8-flash',
        'gemini-3.6-flash',
        'gemini-3.5-flash-lite',
        'gemini-flash-latest',
        'gemini-2.5-flash',
      ];
      const uniqueModels = [...new Set(candidateModels.filter(Boolean))];

      for (const modelId of uniqueModels) {
        try {
          const t0 = Date.now();
          const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelId}:generateContent?key=${geminiKey}`;
          const useThinkingConfig = !this.thinkingConfigUnsupported.has(modelId);
          const buildGenConfig = (withThinking: boolean) => ({
            responseMimeType: 'application/json',
            temperature: 0.2,
            maxOutputTokens: geminiMaxTokens,
            ...(withThinking ? { thinkingConfig: { thinkingBudget: 0 } } : {}),
          });

          let res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              systemInstruction: { parts: [{ text: systemPrompt }] },
              contents: [{ role: 'user', parts: [{ text: userPrompt }] }],
              generationConfig: buildGenConfig(useThinkingConfig),
            }),
          });

          // If endpoint does not support thinkingConfig (HTTP 400), remember and retry immediately without it
          if (res.status === 400 && useThinkingConfig) {
            this.thinkingConfigUnsupported.add(modelId);
            res = await fetch(url, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                systemInstruction: { parts: [{ text: systemPrompt }] },
                contents: [{ role: 'user', parts: [{ text: userPrompt }] }],
                generationConfig: buildGenConfig(false),
              }),
            });
          }

          if (res.ok) {
            const data: any = await res.json();
            const text = data?.candidates?.[0]?.content?.parts?.[0]?.text || null;
            if (text) {
              const dt = Date.now() - t0;
              this.cachedGeminiModelId = modelId;
              console.log(`[System 2 Gemini] Model "${modelId}" responded in ${dt}ms`);
              logVerbose('LLM_RESPONSE', `Received response from Gemini model "${modelId}" in ${dt}ms`, text);
              return text;
            }
          }
        } catch {
          // Try next candidate model
        }
      }
    }

    // 2. Try OpenAI if OPENAI_API_KEY is provided
    if ((provider === 'openai' || provider === 'auto') && process.env.OPENAI_API_KEY) {
      try {
        const res = await fetch('https://api.openai.com/v1/chat/completions', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
          },
          body: JSON.stringify({
            model: process.env.LLM_MODEL || 'gpt-4o-mini',
            messages: [
              { role: 'system', content: systemPrompt },
              { role: 'user', content: userPrompt },
            ],
            response_format: { type: 'json_object' },
            temperature: 0.2,
            frequency_penalty: 0.2,
          }),
        });
        if (res.ok) {
          const data: any = await res.json();
          return data?.choices?.[0]?.message?.content || null;
        }
      } catch {
        // Fall through
      }
    }

    // 3. Try local Ollama SLM if running locally
    if (provider === 'ollama' || (provider === 'auto' && !apiKey)) {
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 1800);
        const res = await fetch(`${CONFIG.llm.ollamaUrl}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: controller.signal,
          body: JSON.stringify({
            model: CONFIG.llm.ollamaModel,
            stream: false,
            format: 'json',
            options: {
              temperature: 0.2,
              repeat_penalty: 1.02,
            },
            messages: [
              { role: 'system', content: systemPrompt },
              { role: 'user', content: userPrompt },
            ],
          }),
        });
        clearTimeout(timeout);
        if (res.ok) {
          const data: any = await res.json();
          return data?.message?.content || null;
        }
      } catch {
        // Fall through
      }
    }

    // 4. Try RTX 5060 Local GPU Neural Daemon as automatic offline fallback
    if (process.env.NEURAL_DISABLE_LOCAL_SLM !== 'true') {
      const gpuOut = await sharedNeuralBridge.generateLocalSLMResponse(
        compactSystemPrompt || systemPrompt,
        compactUserPrompt || userPrompt,
        localMaxTokens,
        undefined,
        undefined,
        undefined,
        undefined,
        botId
      );
      if (gpuOut && gpuOut.content) {
        return gpuOut.content;
      }
    }

    return null;
  }

  /**
   * Zero-Latency Fast-Path Intent Router (<1ms):
   * Handles clear, single-step direct gameplay commands immediately without waiting for LLM token generation.
   */
  private tryFastIntentDecision(
    speaker: string,
    msg: string,
    state: SanitizedWorldState
  ): CognitiveDecision | null {
    const lower = msg.toLowerCase().trim();
    // Do not fast-route complex multi-step chains ("and then", "after that", "armor", "custom", "synthesize")
    if (
      lower.includes(' and ') ||
      lower.includes(' then ') ||
      lower.includes('after') ||
      lower.includes('armor') ||
      lower.includes('custom') ||
      lower.includes('synthesize') ||
      lower.includes('script')
    ) {
      return null;
    }

    if (
      lower === 'hello' ||
      lower === 'hi' ||
      lower === 'hey' ||
      lower === 'yo' ||
      lower === 'sup'
    ) {
      return {
        thought: `Fast-path: Greeting ${speaker}.`,
        speechResponse: `Hey ${speaker}! Ready whenever you are—tell me what to gather, build, or synthesize!`,
        goalCompleted: true,
      };
    }

    if (
      lower === 'stop' ||
      lower === 'halt' ||
      lower === 'cancel' ||
      lower === 'quit' ||
      lower.includes('stop moving') ||
      lower.includes('quit that') ||
      lower.includes('stop that') ||
      lower.includes('cancel that')
    ) {
      return {
        thought: 'Fast-path: Player requested immediate stop.',
        speechResponse: 'Stopping right now and clearing my current task!',
        goalCompleted: true,
        toolCall: { name: 'stop_current_action', arguments: {} },
      };
    }

    if (
      lower === 'follow me' ||
      lower === 'can you follow me' ||
      lower === 'follow' ||
      lower.startsWith('follow me')
    ) {
      return {
        thought: `Fast-path: Following ${speaker}.`,
        speechResponse: `On it ${speaker}, following you!`,
        goalCompleted: true,
        toolCall: { name: 'follow_player', arguments: { playerName: speaker, distance: 3 } },
      };
    }

    if (
      lower === 'come here' ||
      lower === 'can you come here' ||
      lower === 'come to me' ||
      lower === 'come back' ||
      lower.includes('come back to where i am') ||
      lower.includes('come over here') ||
      lower.includes('get over here')
    ) {
      return {
        thought: `Fast-path: Walking to ${speaker}.`,
        speechResponse: `Coming right over, ${speaker}!`,
        goalCompleted: true,
        toolCall: { name: 'come_here', arguments: { playerName: speaker } },
      };
    }

    // Match dropped item recovery: e.g. "pick up your stuff nearby", "collect dropped items", "grab your gear"
    if (
      /\b(pick up|collect|grab|recover|get)\b/.test(lower) &&
      /\b(stuff|dropped|drops|items|loot|gear)\b/.test(lower)
    ) {
      return {
        thought: 'Fast-path: Collecting nearby dropped items and re-equipping gear.',
        speechResponse: 'On it! Picking up the dropped items nearby.',
        goalCompleted: true,
        toolCall: { name: 'collect_dropped_items', arguments: { radius: 32, maxItems: 15 } },
      };
    }

    // Match eating food: e.g. "eat food", "I meant, eat food if you have to", "eat something"
    if (
      /\b(eat|consume)\b/.test(lower) &&
      (lower.includes('food') ||
        lower.includes('beef') ||
        lower.includes('steak') ||
        lower.includes('bread') ||
        lower.includes('apple') ||
        lower.includes('something') ||
        lower === 'eat')
    ) {
      return {
        thought: 'Fast-path: Eating food from inventory.',
        speechResponse: 'Eating some food right now!',
        goalCompleted: true,
        toolCall: { name: 'eat_food', arguments: {} },
      };
    }

    // Match equipping shield or armor: e.g. "i gave you a shield, can you equip it?"
    if (/\bequip\b/.test(lower) && lower.includes('shield')) {
      return {
        thought: 'Fast-path: Equipping shield into off-hand.',
        speechResponse: 'Equipping the shield now!',
        goalCompleted: true,
        toolCall: { name: 'equip_armor_or_item', arguments: { itemName: 'shield' } },
      };
    }

    // Match urgent 1-word mob callouts: e.g. "creeper!", "zombie!", "spider!", "enderman!"
    const urgentMobCallout = lower.match(
      /^(creeper|zombie|spider|skeleton|enderman|witch|blaze|wither_skeleton|wither skeleton)!*$/
    );
    if (urgentMobCallout) {
      const targetType = urgentMobCallout[1].replace(/\s+/g, '_');
      return {
        thought: `Fast-path: Urgent callout for ${targetType}! Engaging immediately.`,
        speechResponse: `On it! Taking out the ${targetType}!`,
        goalCompleted: true,
        toolCall: { name: 'engage_combat', arguments: { targetType, maxRange: 32, count: 1 } },
      };
    }

    // Match direct animal taming: e.g. "tame the wolf", "can you tame that wolf", "tame a cat"
    if (/\btame\b/.test(lower) && /\b(wolf|wolves|dog|dogs|cat|cats|ocelot|parrot)\b/.test(lower)) {
      let animalType = 'wolf';
      if (lower.includes('cat')) animalType = 'cat';
      else if (lower.includes('ocelot')) animalType = 'ocelot';
      else if (lower.includes('parrot')) animalType = 'parrot';
      return {
        thought: `Fast-path: Taming nearby ${animalType}.`,
        speechResponse: `On it! Going to tame the ${animalType}.`,
        goalCompleted: true,
        toolCall: { name: 'tame_animal', arguments: { animalType } },
      };
    }

    // Match direct combat & hunting: e.g. "kill the nearest cow", "kill the nearest like 5 cows", "take care of the 5 spiders", "kill the creeper"
    if (
      /\b(kill|attack|fight|hunt|slay|take care of|eliminate)\b/.test(lower) &&
      /\b(cow|cows|sheep|pig|pigs|chicken|chickens|spider|spiders|cave_spider|zombie|zombies|skeleton|skeletons|wither_skeleton|wither skeleton|wither skeletons|creeper|creepers|enderman|witch|slime|pillager|blaze|hostile|hostiles|mob|mobs|monster|monsters)\b/.test(
        lower
      )
    ) {
      let targetType = 'hostile';
      if (lower.includes('wither skeleton') || lower.includes('wither_skeleton')) {
        targetType = 'wither_skeleton';
      } else if (lower.includes('cave spider') || lower.includes('cave_spider')) {
        targetType = 'cave_spider';
      } else {
        const mobMatch = lower.match(
          /\b(cow|cows|sheep|pig|pigs|chicken|chickens|spider|spiders|zombie|zombies|skeleton|skeletons|creeper|creepers|enderman|witch|slime|pillager|blaze|hostile|hostiles|mob|mobs|monster|monsters)\b/
        );
        if (mobMatch) {
          targetType = mobMatch[1]
            .replace(/^(hostiles|mobs|monsters|mob|monster)$/, 'hostile')
            .replace(/s$/, '');
        }
      }

      const numMatch = lower.match(/\b(\d+)\b/);
      const count = numMatch ? Math.min(16, Math.max(1, parseInt(numMatch[1], 10))) : 1;

      return {
        thought: `Fast-path: Engaging ${count}x ${targetType} for ${speaker}.`,
        speechResponse:
          count > 1
            ? `On it! Hunting ${count} ${targetType}s right now!`
            : `On it! Engaging the nearest ${targetType}!`,
        goalCompleted: true,
        toolCall: {
          name: 'engage_combat',
          arguments: { targetType, maxRange: 36, count },
        },
      };
    }

    // Match direct resource gathering: e.g. "get like 10 logs of any type", "get a stack of sand", "mine 20 stone"
    if (
      /\b(get|gather|mine|chop|collect|dig up|harvest)\b/.test(lower) &&
      /\b(log|logs|wood|sand|dirt|stone|cobble|cobblestone|coal|iron)\b/.test(lower)
    ) {
      let blockType = 'log';
      if (lower.includes('oak')) blockType = 'oak_log';
      else if (lower.includes('spruce')) blockType = 'spruce_log';
      else if (lower.includes('birch')) blockType = 'birch_log';
      else if (lower.includes('sand')) blockType = 'sand';
      else if (lower.includes('dirt')) blockType = 'dirt';
      else if (lower.includes('stone') || lower.includes('cobble')) blockType = 'stone';
      else if (lower.includes('coal')) blockType = 'coal_ore';
      else if (lower.includes('iron')) blockType = 'iron_ore';

      const numMatch = lower.match(/\b(\d+)\b/);
      let count = numMatch ? Math.min(64, Math.max(1, parseInt(numMatch[1], 10))) : 8;
      if (lower.includes('stack')) count = 64;

      return {
        thought: `Fast-path: Harvesting ${count} ${blockType} for ${speaker}.`,
        speechResponse: `On it! Gathering ${count} ${blockType.replace(/_/g, ' ')} right now.`,
        goalCompleted: true,
        toolCall: { name: 'gather_resource', arguments: { blockType, count } },
      };
    }

    if (lower.includes('neural scan') || lower.includes('chunk scan') || lower.includes('gpu scan')) {
      return {
        thought: 'Fast-path: Running RTX 5060 64x64x64 volumetric neural chunk scan.',
        speechResponse: 'Running a 3D neural chunk scan on the GPU!',
        goalCompleted: true,
        toolCall: { name: 'neural_chunk_scan', arguments: { radius: 32 } },
      };
    }

    return null;
  }

  /**
   * Returns the compact state-grounded system prompt (including Inventory, Gear & Durability,
   * Workstations Nearby, Immediately Craftable, Smeltable, Resource Clusters, and 360° Entities)
   * so RL replay buffers and QLoRA training trajectories preserve the exact live sensory state.
   */
  public buildGroundedTrainingSystemPrompt(
    tools: MCPToolSchema[],
    state: SanitizedWorldState,
    stepHistory?: ExecutedStepRecord[]
  ): string {
    return this.buildCompactLocalSystemPrompt(tools, state, [], stepHistory);
  }

  /**
   * Ultra-compact system prompt (~320 tokens vs ~3,800 tokens) tailored specifically for the
   * Local RTX 5060 SLM (Qwen2.5-Coder) so prefill + generation completes in ~1.8 - 2.5 seconds.
   */
  private buildCompactLocalSystemPrompt(
    tools: MCPToolSchema[],
    state: SanitizedWorldState,
    memories: string[],
    stepHistory?: ExecutedStepRecord[]
  ): string {
    const aff = computeGroundedAffordances(state);
    const invShort =
      (state.inventory || []).map((i: any) => `${i.count}x ${i.name}`).join(', ') || 'empty';
    const entitiesShort =
      (state.visibleEntities || [])
        .slice(0, 6)
        .map((e: any) => `${e.username ? `${e.username}(player)` : e.name}(${e.distance.toFixed(1)}m)`)
        .join(', ') || 'none';
    const blocksShort =
      (state.nearbyNotableBlocks || [])
        .slice(0, 6)
        .map((b: any) => `${b.name}@(${b.position.x},${b.position.y},${b.position.z})`)
        .join(', ') || 'none';
    const clustersShort =
      (state.terrainContext?.resourceClusters || [])
        .slice(0, 5)
        .map((c: any) => `${c.blockType}(${c.totalNearby}x,${c.nearestDistance}m ${c.nearestDirection})`)
        .join(', ') || 'none';
    const toolLines = tools
      .map((t) => {
        const params = Object.keys(t.inputSchema?.properties || {}).join(', ');
        return `- ${t.name}(${params})`;
      })
      .join('\n');

    const historyString =
      stepHistory && stepHistory.length > 0
        ? stepHistory
            .slice(-3)
            .map((s) => {
              if (s.success) return `${s.toolName}(OK)`;
              const failReason = (s as any).summary ? `: ${(s as any).summary.slice(0, 55)}` : '';
              return `${s.toolName}(FAIL${failReason})`;
            })
            .join(', ')
        : 'none';

    return `You are NeuroAgent, an autonomous Minecraft AI companion.
State: Pos(${Math.round(state.position.x)},${Math.round(state.position.y)},${Math.round(state.position.z)}) HP:${state.health}/20 Food:${state.food}/20 Held:${state.heldItem || 'none'}
Gear & Durability: ${aff.equippedSummary.shortString}
Inventory: [${invShort}]
Workstations Nearby: [${aff.nearbyWorkstations.join(', ') || 'none'}] | Immediately Craftable: [${aff.craftableNow.join(', ') || 'none'}] | Smeltable: [${aff.smeltableNow ? aff.smeltableItems.join(', ') : 'none'}]
Resource Clusters: [${clustersShort}]
Nearby Entities (360°): [${entitiesShort}]
Nearby Blocks: [${blocksShort}]
Recent Actions: [${historyString}]
Tools:
${toolLines}
Respond ONLY with valid JSON.`;
  }

  private buildDecisionSystemPrompt(
    tools: MCPToolSchema[],
    state: SanitizedWorldState,
    memories: string[]
  ): string {
    const toolSummaries = tools.map((t) => ({
      name: t.name,
      description: t.description,
      parameters: t.inputSchema.properties,
      synthesized: Boolean(t.isSynthesized),
    }));

    const invFormatted = (state.inventory || [])
      .map((i: any) => {
        if (i.durability !== undefined && i.maxDurability !== undefined) {
          return `${i.count}x ${i.name} (durability: ${i.durability}/${i.maxDurability}, ${i.durabilityPercent}%)`;
        }
        return `${i.count}x ${i.name}`;
      })
      .join(', ');

    // STATIC PREFIX FIRST (Maximizes Gemini Implicit Prefix Caching across consecutive turns)
    return `You are NeuroAgent, an autonomous, human-like Minecraft companion player capable of executing both immediate commands and long-horizon multi-step survival goals from start to finish.

Core Execution Rules:
- COMBAT & HUNTING: Always use "engage_combat" to attack, hunt, or kill BOTH hostile mobs (zombie, skeleton, spider, creeper, wither_skeleton) AND passive animals (cow, sheep, pig, chicken). Pass "targetType" (e.g. "cow", "spider", "creeper", "wither_skeleton", "hostile") and "count" (e.g. count: 5 when asked to kill 5 cows or 5 spiders) so all targets are hunted and their drops collected in a SINGLE tool call! NEVER synthesize a separate hunt_animal tool.
- BUILT-IN SURVIVAL SKILLS: Use "collect_dropped_items" to pick up dropped items/gear on the ground, "tame_animal" to tame wolves/cats/parrots, and "pillar_up" to pillar vertically.
- PLAYER USERNAMES: When calling "come_here", "follow_player", or "give_item_to_player", always pass the actual player username (e.g. from the chat speaker or entity username), NEVER the literal word "player".
- CONCISE OUTPUT: Keep "thought" to 1 short tactical sentence (<15 words). Only include "planSteps" when starting a brand-new 3+ step goal.

Output JSON Schema (CognitiveDecision):
{
  "thought": "Concise 1-sentence tactical reasoning (<15 words)",
  "speechResponse": "Optional natural in-game chat message (keep concise; omit on routine intermediate loop steps)",
  "activeGoal": "String describing the overarching goal if multi-step",
  "goalCompleted": false, // Set false when more steps are needed after this toolCall; set true ONLY when the entire goal is finished!
  "planSteps": ["Step 1...", "Step 2..."],
  "rememberLandmark": { "name": "home_base", "position": { "x": 0, "y": 64, "z": 0 } },
  "toolCall": { "name": "tool_name", "arguments": { ... } },
  "synthesizeRequest": { // ONLY use if no existing tool can perform the required step
    "desiredToolName": "snake_case_name",
    "taskDescription": "Detailed specification",
    "requiredParameters": { "param1": "type" },
    "triggeringAsk": "request"
  }
}

Registered MCP Tools:
${JSON.stringify(toolSummaries)}

Current Embodied World & Terrain State:
- Position: (${state.position.x}, ${state.position.y}, ${state.position.z})
- Health: ${state.health}/20, Food: ${state.food}/20, Held Item: ${state.heldItem || 'empty'}
- Equipped Armor & Off-Hand: ${JSON.stringify(state.equippedArmor || {})}
- Inventory: [${invFormatted || 'empty'}]
- Terrain & Tactile Context: ${JSON.stringify(
      state.terrainContext
        ? {
            timeOfDay: state.terrainContext.timeOfDay,
            canSeeSky: state.terrainContext.canSeeSky,
            blockBelowFeet: state.terrainContext.blockBelowFeet,
            depthToStoneBlocksBelowFeet: state.terrainContext.depthToStoneBlocks,
            adjacentBlocks: state.terrainContext.adjacentBlocks,
          }
        : {}
    )}
- 360° Nearby Resource Clusters (within 22m radius):
${JSON.stringify(state.terrainContext?.resourceClusters || [])}
- Nearby & Visible Entities (360° Radar within 32m): ${JSON.stringify(
      state.visibleEntities.map((e: any) => ({
        name: e.username ? `${e.username} (player)` : e.name,
        type: e.type,
        dist: e.distance,
        inFov: e.inFov,
      }))
    )}
- Stored Episodic Memories & Landmarks: ${JSON.stringify(memories)}`;
  }

  /**
   * Deterministic Autonomous Tech-Tree & Goal Continuation Engine
   * (Used if offline or if an API call times out mid-loop so the agent NEVER gets stuck!)
   */
  private fallbackGoalContinuation(
    activeGoal: string,
    stepHistory: ExecutedStepRecord[],
    state: SanitizedWorldState
  ): CognitiveDecision {
    const goalLower = activeGoal.toLowerCase();
    const inv = state.inventory;
    const countItem = (substr: string) =>
      inv
        .filter((i) => i.name.toLowerCase().includes(substr))
        .reduce((acc, i) => acc + i.count, 0);

    const lastStep = stepHistory[stepHistory.length - 1];

    // If the goal involves iron armor / iron progression:
    if (goalLower.includes('iron')) {
      const hasHelmet = countItem('iron_helmet') > 0;
      const hasChest = countItem('iron_chestplate') > 0;
      const hasLegs = countItem('iron_leggings') > 0;
      const hasBoots = countItem('iron_boots') > 0;

      if (hasHelmet && hasChest && hasLegs && hasBoots) {
        if (lastStep?.toolName !== 'equip_armor_or_item') {
          return {
            thought: 'All 4 iron armor pieces are crafted! Equipping full iron armor now.',
            speechResponse: 'I have crafted the full set of iron armor! Equipping it now.',
            activeGoal,
            goalCompleted: false,
            toolCall: { name: 'equip_armor_or_item', arguments: { itemName: 'all_armor' } },
          };
        }
        return {
          thought: 'Full iron armor is crafted and equipped. Goal complete!',
          speechResponse: 'Mission accomplished! I now have full iron armor equipped.',
          activeGoal,
          goalCompleted: true,
        };
      }

      const ironIngots = countItem('iron_ingot');
      if (!hasChest && ironIngots >= 8) {
        return {
          thought: 'Have >= 8 iron ingots, crafting iron_chestplate.',
          activeGoal,
          goalCompleted: false,
          toolCall: { name: 'craft_item', arguments: { itemName: 'iron_chestplate', count: 1 } },
        };
      }
      if (!hasLegs && ironIngots >= 7) {
        return {
          thought: 'Have >= 7 iron ingots, crafting iron_leggings.',
          activeGoal,
          goalCompleted: false,
          toolCall: { name: 'craft_item', arguments: { itemName: 'iron_leggings', count: 1 } },
        };
      }
      if (!hasHelmet && ironIngots >= 5) {
        return {
          thought: 'Have >= 5 iron ingots, crafting iron_helmet.',
          activeGoal,
          goalCompleted: false,
          toolCall: { name: 'craft_item', arguments: { itemName: 'iron_helmet', count: 1 } },
        };
      }
      if (!hasBoots && ironIngots >= 4) {
        return {
          thought: 'Have >= 4 iron ingots, crafting iron_boots.',
          activeGoal,
          goalCompleted: false,
          toolCall: { name: 'craft_item', arguments: { itemName: 'iron_boots', count: 1 } },
        };
      }

      // Check if we have raw_iron to smelt
      const rawIron = countItem('raw_iron') + countItem('iron_ore');
      const fuel = countItem('coal') + countItem('_planks') + countItem('_log');
      if (rawIron >= 4 && fuel >= 1) {
        return {
          thought: `Have ${rawIron} raw iron and fuel. Smelting into iron_ingot.`,
          activeGoal,
          goalCompleted: false,
          toolCall: { name: 'smelt_item', arguments: { inputItem: 'raw_iron', count: rawIron } },
        };
      }

      // Check if we have a stone_pickaxe (required to mine iron_ore!)
      const hasStonePick =
        countItem('stone_pickaxe') > 0 ||
        countItem('iron_pickaxe') > 0 ||
        countItem('diamond_pickaxe') > 0;
      if (hasStonePick) {
        if (lastStep && lastStep.toolName === 'gather_resource' && !lastStep.success) {
          return {
            thought: 'No exposed iron_ore on surface; digging staircase mine to expose underground iron and coal.',
            activeGoal,
            goalCompleted: false,
            toolCall: {
              name: 'explore_or_dig_for_ores',
              arguments: { mode: 'staircase_mine', steps: 6 },
            },
          };
        }
        return {
          thought: 'Have stone pickaxe; searching and mining iron_ore.',
          activeGoal,
          goalCompleted: false,
          toolCall: { name: 'gather_resource', arguments: { blockType: 'iron_ore', count: 6 } },
        };
      }

      // Need stone_pickaxe (requires 3 cobblestone + 2 sticks)
      const cobble = countItem('cobblestone') + countItem('cobbled_deepslate');
      if (cobble >= 11) {
        return {
          thought: `Have ${cobble} cobblestone. Crafting stone_pickaxe!`,
          activeGoal,
          goalCompleted: false,
          toolCall: { name: 'craft_item', arguments: { itemName: 'stone_pickaxe', count: 1 } },
        };
      }

      // Check if we have wooden_pickaxe to mine stone
      const hasWoodPick = countItem('wooden_pickaxe') > 0;
      if (hasWoodPick) {
        if (lastStep && lastStep.toolName === 'gather_resource' && !lastStep.success) {
          return {
            thought: 'Digging staircase down through dirt layer to reach stone.',
            activeGoal,
            goalCompleted: false,
            toolCall: {
              name: 'explore_or_dig_for_ores',
              arguments: { mode: 'staircase_mine', steps: 5 },
            },
          };
        }
        return {
          thought: 'Using wooden_pickaxe to mine 11 stone/cobblestone for stone_pickaxe and furnace.',
          activeGoal,
          goalCompleted: false,
          toolCall: { name: 'gather_resource', arguments: { blockType: 'stone', count: 11 } },
        };
      }

      // Need wooden_pickaxe (requires planks + sticks + crafting_table)
      const woodCount = countItem('_log') * 4 + countItem('_planks');
      if (woodCount >= 5) {
        return {
          thought: 'Crafting wooden_pickaxe (auto-crafting sticks and placing crafting_table).',
          activeGoal,
          goalCompleted: false,
          toolCall: { name: 'craft_item', arguments: { itemName: 'wooden_pickaxe', count: 1 } },
        };
      }

      return {
        thought: 'Need wood logs to start tool progression.',
        activeGoal,
        goalCompleted: false,
        toolCall: { name: 'gather_resource', arguments: { blockType: 'log', count: 4 } },
      };
    }

    return {
      thought: 'Completed active task sequence.',
      activeGoal,
      goalCompleted: true,
    };
  }

  private fallbackHeuristicDecision(
    speaker: string,
    msg: string,
    state: SanitizedWorldState,
    availableTools: MCPToolSchema[]
  ): CognitiveDecision {
    const lower = msg.toLowerCase().trim();

    // Check if the user is asking for a long-term iron armor / progression task
    if (lower.includes('iron armor') || lower.includes('long term') || lower.includes('full iron')) {
      return {
        thought: `Player ${speaker} assigned long-term goal: "${msg}". Saving current coordinates (${state.position.x}, ${state.position.y}, ${state.position.z}) and initiating autonomous survival progression loop.`,
        speechResponse: `Understood! Remembering our coordinates at (${Math.round(state.position.x)}, ${Math.round(state.position.y)}, ${Math.round(state.position.z)}) and starting the autonomous progression to full iron armor!`,
        activeGoal: msg,
        goalCompleted: false,
        rememberLandmark: {
          name: 'start_base',
          position: state.position,
        },
        toolCall: this.fallbackGoalContinuation(msg, [], state).toolCall,
      };
    }

    for (const tool of availableTools) {
      if (tool.isSynthesized) {
        const nameWords = tool.name.split('_').filter((w: string) => w.length > 2);
        const matches = nameWords.filter((w: string) => lower.includes(w));
        if (matches.length >= Math.max(1, nameWords.length - 1)) {
          return {
            thought: `Matched previously synthesized MCP tool "${tool.name}".`,
            speechResponse: `Running my custom ${tool.name.replace(/_/g, ' ')} skill!`,
            goalCompleted: true,
            toolCall: { name: tool.name, arguments: { speaker } },
          };
        }
      }
    }

    if (lower.includes('stop') || lower.includes('halt') || lower.includes('wait') || lower.includes('cancel')) {
      return {
        thought: 'User requested to stop current action.',
        speechResponse: 'Stopping right here and clearing my active goal!',
        goalCompleted: true,
        toolCall: { name: 'stop_current_action', arguments: {} },
      };
    }

    if (lower.includes('follow') || lower.includes('with me')) {
      return {
        thought: `Following player ${speaker}.`,
        speechResponse: `Got it ${speaker}, I'm right behind you!`,
        goalCompleted: true,
        toolCall: { name: 'follow_player', arguments: { playerName: speaker, distance: 3 } },
      };
    }

    if (lower.includes('come here') || lower.includes('come to me') || lower === 'come') {
      return {
        thought: `Moving to ${speaker}'s location.`,
        speechResponse: `On my way to you, ${speaker}!`,
        goalCompleted: true,
        toolCall: { name: 'come_here', arguments: { playerName: speaker } },
      };
    }

    if (lower.includes('wood') || lower.includes('log') || lower.includes('tree') || lower.includes('mine') || lower.includes('gather') || lower.includes('cobble') || lower.includes('stone') || lower.includes('iron') || lower.includes('coal')) {
      let blockType = 'log';
      if (lower.includes('birch')) blockType = 'birch_log';
      else if (lower.includes('spruce')) blockType = 'spruce_log';
      else if (lower.includes('stone') || lower.includes('cobble')) blockType = 'stone';
      else if (lower.includes('iron')) blockType = 'iron_ore';
      else if (lower.includes('coal')) blockType = 'coal_ore';
      else if (lower.includes('dirt')) blockType = 'dirt';
      else if (lower.includes('sand')) blockType = 'sand';

      const numMatch = lower.match(/\b(\d+)\b/);
      const count = numMatch ? Math.min(32, parseInt(numMatch[1], 10)) : 4;

      return {
        thought: `Gathering ${count} ${blockType}.`,
        speechResponse: `Sure thing! I'll go harvest ${count} ${blockType.replace(/_/g, ' ')}.`,
        goalCompleted: true,
        toolCall: { name: 'gather_resource', arguments: { blockType, count } },
      };
    }

    if (
      lower.includes('house') ||
      lower.includes('shelter') ||
      lower.includes('cabin') ||
      (lower.includes('build') && !lower.includes('pillar') && !lower.includes('tower'))
    ) {
      return {
        thought: 'Constructing survival shelter/house with walls, roof, doorway, and ground-anchored foundation.',
        speechResponse: "On it! I'll scan the ground topology and build a simple house right here.",
        goalCompleted: true,
        toolCall: { name: 'build_simple_house', arguments: { width: 5, length: 5, height: 3 } },
      };
    }

    if (
      lower.includes('scan') &&
      (lower.includes('area') || lower.includes('terrain') || lower.includes('ground') || lower.includes('topology'))
    ) {
      return {
        thought: 'Scanning 3D area terrain topology and ground elevations.',
        speechResponse: 'Scanning the terrain around us now!',
        goalCompleted: true,
        toolCall: { name: 'scan_area', arguments: { radius: 8 } },
      };
    }

    if (lower.includes('craft') || lower.includes('make a ')) {
      let itemName = 'planks';
      if (lower.includes('pickaxe')) itemName = lower.includes('stone') ? 'stone_pickaxe' : 'wooden_pickaxe';
      else if (lower.includes('sword')) itemName = lower.includes('stone') ? 'stone_sword' : 'wooden_sword';
      else if (lower.includes('stick')) itemName = 'stick';
      else if (lower.includes('table') || lower.includes('workbench')) itemName = 'crafting_table';
      else if (lower.includes('furnace')) itemName = 'furnace';
      else if (lower.includes('torch')) itemName = 'torch';

      return {
        thought: `Crafting requested item: ${itemName}.`,
        speechResponse: `Crafting ${itemName.replace(/_/g, ' ')} now!`,
        goalCompleted: true,
        toolCall: { name: 'craft_item', arguments: { itemName, count: 1 } },
      };
    }

    if (lower.includes('attack') || lower.includes('fight') || lower.includes('kill') || lower.includes('defend') || lower.includes('protect')) {
      return {
        thought: 'Engaging nearby hostile mobs using 1.9+ combat cooldowns.',
        speechResponse: 'Drawing weapon! I will clear out the mobs.',
        goalCompleted: true,
        toolCall: { name: 'engage_combat', arguments: { targetType: 'hostile', maxRange: 16 } },
      };
    }

    if (lower.includes('give') || lower.includes('drop') || lower.includes('toss')) {
      const words = lower.replace(/[^a-z0-9_ ]/g, '').split(/\s+/);
      const itemHint = words[words.length - 1] || 'any';
      return {
        thought: `Giving ${itemHint} to ${speaker}.`,
        speechResponse: `Here you go, tossing ${itemHint} to you!`,
        goalCompleted: true,
        toolCall: { name: 'give_item_to_player', arguments: { playerName: speaker, itemName: itemHint } },
      };
    }

    if (lower.includes('status') || lower.includes('inventory') || lower.includes('what do you see') || lower.includes('look around')) {
      return {
        thought: 'Inspecting surroundings and reporting status.',
        speechResponse: `HP: ${state.health}/20, Food: ${state.food}/20. I see ${state.visibleEntities.length} entities in my FOV and have ${state.inventory.length} item stacks.`,
        goalCompleted: true,
        toolCall: { name: 'inspect_surroundings', arguments: {} },
      };
    }

    const cleanWords = lower
      .replace(/[^a-z0-9 ]/g, '')
      .split(/\s+/)
      .filter((w: string) => !['hey', 'agent', 'neuroagent', 'please', 'can', 'you', 'a', 'the', 'to', 'in', 'for', 'me'].includes(w))
      .slice(0, 3);

    if (cleanWords.length >= 1) {
      const desiredToolName = `custom_${cleanWords.join('_')}`.replace(/[^a-z0-9_]/g, '').slice(0, 36);
      return {
        thought: `Detected novel ask "${msg}". Triggering Self-Synthesizing MCP Tool Factory to create "${desiredToolName}".`,
        speechResponse: `Hold on a sec, let me figure out a custom skill for "${cleanWords.join(' ')}"!`,
        goalCompleted: true,
        synthesizeRequest: {
          desiredToolName,
          taskDescription: `Execute custom player instruction: ${msg}`,
          requiredParameters: { speaker: 'string' },
          triggeringAsk: msg,
        },
      };
    }

    return {
      thought: 'Conversational greeting.',
      speechResponse: `Hey ${speaker}! Tell me any short task or a long-term goal (like getting full iron armor) and I'll execute every step start-to-finish!`,
      goalCompleted: true,
    };
  }

  private fallbackSynthesizeTool(
    desiredToolName: string,
    taskDescription: string,
    triggeringAsk: string
  ): {
    description: string;
    parameters: Record<string, { type: string; description: string; default?: any }>;
    functionBodyCode: string;
    tags: string[];
  } {
    const lower = triggeringAsk.toLowerCase();

    if (lower.includes('dance') || lower.includes('spin') || lower.includes('celebrate') || lower.includes('flip') || lower.includes('party')) {
      return {
        description: `Performs a humanized celebratory dance with smooth spins, jumps, and arm swings (${triggeringAsk})`,
        parameters: {
          durationSeconds: { type: 'number', description: 'Dance duration in seconds', default: 4 },
        },
        tags: ['dance', 'spin', 'celebrate', 'party', 'jump'],
        functionBodyCode: `
          const pos = ctx.bot.entity.position;
          const steps = 8;
          for (let i = 0; i < steps; i++) {
            if (ctx.abortSignal.aborted) break;
            ctx.reportProgress((i + 1) / steps);
            const angle = (i * Math.PI) / 2;
            const lookTarget = new Vec3(pos.x + Math.cos(angle) * 3, pos.y + 1.6 + (i % 2 === 0 ? 0.5 : -0.3), pos.z + Math.sin(angle) * 3);
            await ctx.kinematics.lookAtSmooth(lookTarget, 180);
            ctx.bot.setControlState('jump', i % 2 === 0);
            ctx.bot.setControlState('sneak', i % 2 === 1);
            ctx.bot.swingArm('right');
            await new Promise(r => setTimeout(r, 260));
          }
          ctx.bot.setControlState('jump', false);
          ctx.bot.setControlState('sneak', false);
          return { success: true, summary: 'Finished performing celebratory dance routine!' };
        `,
      };
    }

    if (lower.includes('patrol') || lower.includes('perimeter') || lower.includes('circle') || lower.includes('guard') || lower.includes('scout')) {
      return {
        description: `Patrols a smooth spline perimeter around the current location while scanning for threats (${triggeringAsk})`,
        parameters: {
          radius: { type: 'number', description: 'Patrol radius in blocks', default: 5 },
        },
        tags: ['patrol', 'perimeter', 'circle', 'guard', 'scout'],
        functionBodyCode: `
          const r = Number(args.radius) || 5;
          const origin = ctx.bot.entity.position.clone();
          const waypoints = [
            new Vec3(origin.x + r, origin.y, origin.z),
            new Vec3(origin.x + r, origin.y, origin.z + r),
            new Vec3(origin.x, origin.y, origin.z + r),
            new Vec3(origin.x, origin.y, origin.z),
          ];
          ctx.reportProgress(0.25);
          await ctx.kinematics.moveAlongSpline(waypoints, true);
          ctx.reportProgress(1.0);
          return { success: true, summary: 'Completed perimeter patrol circuit.' };
        `,
      };
    }

    if (lower.includes('build') || lower.includes('tower') || lower.includes('pillar') || lower.includes('hut') || lower.includes('shelter') || lower.includes('wall')) {
      return {
        description: `Builds a block structure near the agent using available placeable blocks in inventory (${triggeringAsk})`,
        parameters: {
          height: { type: 'number', description: 'Height or count of blocks to place', default: 3 },
        },
        tags: ['build', 'tower', 'pillar', 'hut', 'shelter', 'wall', 'place'],
        functionBodyCode: `
          const placeables = ['cobblestone', 'dirt', 'oak_planks', 'spruce_planks', 'birch_planks', 'stone', 'sandstone', 'glass'];
          let chosen = null;
          for (const item of placeables) {
            if (await ctx.kinematics.equipItemHumanized(item, 'hand')) {
              chosen = item;
              break;
            }
          }
          if (!chosen) {
            return { success: false, summary: 'No placeable building blocks (cobble, dirt, planks) found in inventory!' };
          }
          const base = ctx.bot.entity.position.floored().offset(2, 0, 0);
          let placed = 0;
          for (let dx = 0; dx < 3; dx++) {
            if (ctx.abortSignal.aborted) break;
            const targetX = base.x + dx;
            const targetZ = base.z;
            const groundY = ctx.world.getGroundHeight(targetX, targetZ);
            const targetPos = new Vec3(targetX, groundY + 1, targetZ);
            const ok = await ctx.kinematics.safePlaceBlock(targetPos, chosen);
            if (ok) placed++;
            ctx.reportProgress((dx + 1) / 3);
          }
          return { success: true, summary: 'Constructed structure segment using ' + chosen + ' (' + placed + ' blocks placed).' };
        `,
      };
    }

    return {
      description: `Synthesized custom skill for: ${taskDescription}`,
      parameters: {
        speaker: { type: 'string', description: 'Player who requested the skill', default: 'Player' },
      },
      tags: desiredToolName.split('_'),
      functionBodyCode: `
        const startPos = ctx.bot.entity.position.clone();
        ctx.reportProgress(0.3);
        const look1 = new Vec3(startPos.x + 3, startPos.y + 1.6, startPos.z + 2);
        await ctx.kinematics.lookAtSmooth(look1, 220);
        ctx.bot.swingArm('right');
        ctx.reportProgress(0.75);
        await ctx.kinematics.moveAlongSpline([
          new Vec3(startPos.x + 2, startPos.y, startPos.z + 1),
          startPos
        ], true);
        ctx.reportProgress(1.0);
        return { success: true, summary: 'Executed synthesized MCP skill ${desiredToolName}!' };
      `,
    };
  }

  private tryParseJson<T>(raw: string): T | null {
    try {
      // 1. Standard JSON & Markdown code block parser (Primary structured interface)
      const cleaned = raw
        .replace(/^```json\s*/i, '')
        .replace(/^```\s*/i, '')
        .replace(/```\s*$/, '')
        .trim();

      let obj: any = null;
      try {
        obj = JSON.parse(cleaned);
      } catch {
        const first = raw.indexOf('{');
        const last = raw.lastIndexOf('}');
        if (first !== -1 && last > first) {
          try {
            obj = JSON.parse(raw.substring(first, last + 1));
          } catch {}
        }
      }

      if (obj && typeof obj === 'object') {
        // Robust schema normalization: map executedTools array to toolCall
        if (!obj.toolCall && Array.isArray(obj.executedTools) && obj.executedTools.length > 0) {
          const t = obj.executedTools[0];
          obj.toolCall = {
            name: t.name,
            arguments: t.args || t.arguments || {},
          };
        }
        if (!obj.toolCall && obj.tool) {
          obj.toolCall = {
            name: obj.tool,
            arguments: obj.args || obj.arguments || {},
          };
        }
        if (!obj.toolCall && obj.action) {
          obj.toolCall = {
            name: obj.action,
            arguments: obj.args || obj.arguments || {},
          };
        }

        if (obj.toolCall && typeof obj.toolCall.name === 'string') {
          const rawName = obj.toolCall.name.trim();
          const snakeName = rawName
            .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
            .toLowerCase();
          const lowerName = rawName.toLowerCase();
          if (DynamicMCPToolRegistry.TOOL_ALIASES[snakeName]) {
            obj.toolCall.name = DynamicMCPToolRegistry.TOOL_ALIASES[snakeName];
          } else if (DynamicMCPToolRegistry.TOOL_ALIASES[lowerName]) {
            obj.toolCall.name = DynamicMCPToolRegistry.TOOL_ALIASES[lowerName];
          } else {
            obj.toolCall.name = snakeName;
          }
          if (obj.toolCall.arguments && typeof obj.toolCall.arguments === 'object') {
            if (
              !obj.toolCall.arguments.itemName &&
              (obj.toolCall.arguments.recipeName ||
                obj.toolCall.arguments.item ||
                obj.toolCall.arguments.target ||
                obj.toolCall.arguments.armor)
            ) {
              obj.toolCall.arguments.itemName =
                obj.toolCall.arguments.recipeName ||
                obj.toolCall.arguments.item ||
                obj.toolCall.arguments.target ||
                obj.toolCall.arguments.armor;
            }
          }
        }

        return obj as T;
      }

      // 2. Support ChatML / Qwen <tool_call> special token grammar
      if (raw.includes('<tool_call>')) {
        const match = raw.match(/<tool_call>([\s\S]*?)<\/tool_call>/i) || raw.match(/<tool_call>([\s\S]*)/i);
        if (match && match[1]) {
          try {
            const toolObj = JSON.parse(match[1].trim());
            if (toolObj && (toolObj.name || toolObj.arguments)) {
              let toolName = toolObj.name;
              const lower = (toolName || '').toLowerCase().trim();
              if (DynamicMCPToolRegistry.TOOL_ALIASES[lower]) {
                toolName = DynamicMCPToolRegistry.TOOL_ALIASES[lower];
              }
              return {
                thought: `Tool call parsed via ChatML token grammar: ${toolName}`,
                toolCall: {
                  name: toolName,
                  arguments: toolObj.arguments || {},
                },
              } as unknown as T;
            }
          } catch {}
        }
      }

      // 3. Support Native Minecraft VLA Action Tokens (<|act:...|>)
      if (raw.includes('<|act:')) {
        const actMatch = raw.match(/<\|act:([a-zA-Z0-9_]+)\|>/i);
        if (actMatch && actMatch[1]) {
          const actName = actMatch[1];
          const targetMatch = raw.match(/<\|target:([a-zA-Z0-9_]+)\|>/i);
          const itemMatch = raw.match(/<\|item:([a-zA-Z0-9_]+)\|>/i);
          const qtyMatch = raw.match(/<\|qty:(\d+)\|>/i);
          const target = targetMatch ? targetMatch[1] : undefined;
          const item = itemMatch ? itemMatch[1] : undefined;
          const qty = qtyMatch ? parseInt(qtyMatch[1], 10) : 1;

          let mappedToolName = actName;
          const mappedArgs: Record<string, any> = {};

          if (actName === 'kite_combat' || actName === 'melee_attack' || actName === 'shield_block' || actName === 'attack_mob') {
            mappedToolName = 'engage_combat';
            if (target) {
              mappedArgs.targetType = target;
              mappedArgs.targetMob = target;
              mappedArgs.target = target;
            }
          } else if (actName === 'parkour_jump') {
            mappedToolName = 'navigate_parkour_or_bridge';
            mappedArgs.mode = 'parkour';
          } else if (actName === 'bridge_sneak') {
            mappedToolName = 'navigate_parkour_or_bridge';
            mappedArgs.mode = 'bridge';
          } else if (actName === 'till_and_plant' || actName === 'harvest_crops' || actName === 'till_soil') {
            mappedToolName = 'till_and_farm';
            mappedArgs.cropType = 'wheat';
          } else if (actName === 'tame_animal') {
            mappedToolName = 'tame_animal';
            mappedArgs.animalType = target || 'wolf';
          } else if (actName === 'shoot_bow') {
            mappedToolName = 'shoot_bow';
            if (target) mappedArgs.target = target;
          } else if (actName === 'pillar_up') {
            mappedToolName = 'pillar_up';
            mappedArgs.height = 5;
          } else if (actName === 'interact_block') {
            mappedToolName = 'interact_block_or_puzzle';
            mappedArgs.action = 'flip_lever';
          } else if (actName === 'smelt' || actName === 'smelt_item' || actName === 'smelt_iron') {
            mappedToolName = 'smelt_item';
            mappedArgs.itemName = item || target || 'raw_iron';
          } else if (actName === 'equip' || actName === 'equip_armor' || actName === 'equip_item') {
            mappedToolName = 'equip_armor_or_item';
            mappedArgs.itemName = item || target || 'all_armor';
          } else if (actName === 'craft' || actName === 'craft_recipe' || actName === 'craft_item') {
            mappedToolName = 'craft_item';
            mappedArgs.itemName = item || target || 'crafting_table';
            mappedArgs.count = qty || 1;
          } else if (actName.startsWith('craft_')) {
            mappedToolName = 'craft_item';
            mappedArgs.itemName = actName.replace(/^craft_/, '');
            mappedArgs.count = qty || 1;
          } else if (actName === 'mine_wood' || actName === 'chop_wood' || actName === 'gather_wood' || actName === 'gather_resource') {
            mappedToolName = 'gather_resource';
            mappedArgs.blockType = 'log';
            mappedArgs.count = qty || 4;
          } else if (actName === 'mine_stone' || actName === 'mine_cobblestone') {
            mappedToolName = 'gather_resource';
            mappedArgs.blockType = 'stone';
            mappedArgs.count = qty || 3;
          } else if (actName === 'mine_iron') {
            mappedToolName = 'gather_resource';
            mappedArgs.blockType = 'iron_ore';
            mappedArgs.count = qty || 3;
          } else if (actName === 'drop_item' || actName === 'give_to_player') {
            mappedToolName = 'drop_item';
            if (item) mappedArgs.item = item;
            mappedArgs.quantity = qty;
          } else if (actName.includes('mine') || actName.includes('gather') || actName.includes('harvest')) {
            mappedToolName = 'gather_resource';
            mappedArgs.blockType = item || target || 'log';
          }

          const lowerMapped = (mappedToolName || '').toLowerCase().trim();
          if (DynamicMCPToolRegistry.TOOL_ALIASES[lowerMapped]) {
            mappedToolName = DynamicMCPToolRegistry.TOOL_ALIASES[lowerMapped];
          }

          return {
            thought: `Parsed VLA Action Token: ${actName}`,
            toolCall: {
              name: mappedToolName,
              arguments: mappedArgs,
            },
          } as unknown as T;
        }
      }

      // 4. Natural Language Intent Fallback for local SLMs (only if structured formats failed)
      const lowerRaw = raw.toLowerCase();
      if (lowerRaw.includes('wood') || lowerRaw.includes('log') || lowerRaw.includes('chop') || lowerRaw.includes('tree')) {
        return {
          thought: `Natural language intent extracted: ${raw.slice(0, 60)}`,
          toolCall: { name: 'gather_resource', arguments: { blockType: 'log', count: 4 } },
        } as unknown as T;
      }
      if (lowerRaw.includes('craft') && (lowerRaw.includes('table') || lowerRaw.includes('workbench'))) {
        return {
          thought: `Natural language intent extracted: ${raw.slice(0, 60)}`,
          toolCall: { name: 'craft_item', arguments: { itemName: 'crafting_table', count: 1 } },
        } as unknown as T;
      }
      if (lowerRaw.includes('craft') && lowerRaw.includes('pickaxe')) {
        return {
          thought: `Natural language intent extracted: ${raw.slice(0, 60)}`,
          toolCall: { name: 'craft_item', arguments: { itemName: 'wooden_pickaxe', count: 1 } },
        } as unknown as T;
      }
      if (lowerRaw.includes('equip') && (lowerRaw.includes('armor') || lowerRaw.includes('helmet') || lowerRaw.includes('chestplate') || lowerRaw.includes('boots') || lowerRaw.includes('shield'))) {
        return {
          thought: `Natural language intent extracted: ${raw.slice(0, 60)}`,
          toolCall: { name: 'equip_armor_or_item', arguments: { itemName: 'all_armor' } },
        } as unknown as T;
      }
      if (lowerRaw.includes('stone') && (lowerRaw.includes('mine') || lowerRaw.includes('dig') || lowerRaw.includes('harvest'))) {
        return {
          thought: `Natural language intent extracted: ${raw.slice(0, 60)}`,
          toolCall: { name: 'gather_resource', arguments: { blockType: 'stone', count: 3 } },
        } as unknown as T;
      }

      return null;
    } catch {
      return null;
    }
  }
}
