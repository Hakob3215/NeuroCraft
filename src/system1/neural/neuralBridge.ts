import { SanitizedWorldState } from '../../types';
import { CONFIG } from '../../config';
import { logVerbose } from '../../logger';

export interface NeuralReflexOutput {
  aim_delta: { dyaw: number; dpitch: number };
  actions: {
    move_forward: boolean;
    move_back: boolean;
    strafe_left: boolean;
    strafe_right: boolean;
    jump: boolean;
    sprint: boolean;
    sneak: boolean;
    attack: boolean;
    use_shield: boolean;
  };
  action_probs: Record<string, number>;
  hazard_urgency: number;
  latency_ms: number;
}

export interface NeuralChunkScanOutput {
  recommended_build_offset: { dx: number; dy: number; dz: number };
  flatness_confidence: number;
  recommended_ore_offset: { dx: number; dy: number; dz: number };
  ore_richness_score: number;
  cavern_hazard_score: number;
  latency_ms: number;
}

export interface NeuralLLMOutput {
  content: string;
  tokens_generated: number;
  latency_ms: number;
  tokens_per_sec: number;
}

/**
 * High-speed IPC bridge connecting System 1 & System 2 to the RTX 5060 Local Neural Daemon
 * (neural_lab/neural_daemon.py listening on http://127.0.0.1:8765).
 * Automatically caches availability so offline runs incur zero latency overhead.
 */
export class NeuralBridgeClient {
  private readonly baseUrl: string;
  private readonly fetchFn: typeof fetch;
  private isDaemonOnline = false;
  private lastHealthCheck = 0;

  constructor(baseUrl?: string, fetchImpl?: typeof fetch) {
    this.baseUrl = baseUrl || CONFIG.llm.neuralDaemonUrl || 'http://127.0.0.1:8765';
    this.fetchFn = fetchImpl || globalThis.fetch.bind(globalThis);
  }

  public async checkHealth(force = false): Promise<boolean> {
    const now = Date.now();
    if (!force && now - this.lastHealthCheck < 5000 && this.isDaemonOnline) {
      return this.isDaemonOnline;
    }
    this.lastHealthCheck = now;
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 2500);
      const res = await this.fetchFn(`${this.baseUrl}/health`, { signal: controller.signal });
      clearTimeout(timeout);
      this.isDaemonOnline = res.ok;
      return this.isDaemonOnline;
    } catch {
      // If it was previously online, a brief delay during GPU load should not immediately drop daemon status
      if (this.isDaemonOnline) {
        return true;
      }
      this.isDaemonOnline = false;
      return false;
    }
  }

  /**
   * Queries the RTX 5060 VoxelReflexPolicy (~2.0ms GPU inference) for sub-tick combat
   * and hazard motor reflexes.
   */
  public async evaluateReflex(state: SanitizedWorldState): Promise<NeuralReflexOutput | null> {
    if (!(await this.checkHealth())) return null;

    const botPos = state.position || (state as any).botPosition || { x: 0, y: 64, z: 0 };
    const blocksList = state.nearbyNotableBlocks || (state as any).notableBlocks || [];
    const sparseBlocks = blocksList.slice(0, 64).map((b: any) => ({
      dx: Math.round(b.position.x - botPos.x),
      dy: Math.round(b.position.y - botPos.y),
      dz: Math.round(b.position.z - botPos.z),
      name: b.name,
    }));

    const entities = (state.visibleEntities || []).slice(0, 16).map((e: any) => {
      const eName = String(e.name || e.username || '').toLowerCase();
      const isRivalPlayer =
        e.type === 'player' &&
        (eName.includes('opponent') ||
          eName.includes('mirror') ||
          eName.includes('dqn') ||
          eName.includes('rival') ||
          eName.includes('spar')) &&
        !eName.includes('companion') &&
        !eName.includes('friend') &&
        !eName.includes('teammate');

      return {
        rx: e.position.x - botPos.x,
        ry: e.position.y - botPos.y,
        rz: e.position.z - botPos.z,
        vx: e.velocity?.x || 0,
        vy: e.velocity?.y || 0,
        vz: e.velocity?.z || 0,
        distance: e.distance,
        isHostile: e.type === 'hostile' || Boolean(e.isHostile) || isRivalPlayer,
        name: e.name,
        healthNorm: 1.0,
      };
    });

    const hasShield =
      state.equippedArmor?.offHand?.includes('shield') ||
      (state as any).offHandItem?.includes('shield') ||
      (state.inventory || []).some((i) => i.name.includes('shield'));

    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 45); // Strict 45ms sub-tick bound
      const res = await this.fetchFn(`${this.baseUrl}/reflex`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          sparse_blocks: sparseBlocks,
          entities,
          bot_state: {
            health: state.health,
            food: state.food,
            oxygen: state.oxygen,
            onGround: state.onGround,
            isInWater: state.isInWater,
            isInLava: state.isInLava,
            hasShield,
            weaponCooldown: 1.0,
            pitch: state.pitch ?? (state as any).botPitch ?? 0,
            yaw: state.yaw ?? (state as any).botYaw ?? 0,
          },
        }),
      });
      clearTimeout(timeout);
      if (!res.ok) return null;
      return (await res.json()) as NeuralReflexOutput;
    } catch {
      return null;
    }
  }

  /**
   * Queries the RTX 5060 VolumetricChunkAnalyzer (64x64x64 3D Scene Transformer, ~4.8ms)
   * to locate optimal flat building pads, rich ore veins, and subterranean hazards.
   */
  public async evaluateChunkScan(
    sparseBlocks: Array<{ dx: number; dy: number; dz: number; name: string }>
  ): Promise<NeuralChunkScanOutput | null> {
    if (!(await this.checkHealth(true))) return null;

    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 1500);
      const res = await this.fetchFn(`${this.baseUrl}/chunk_scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({ sparse_blocks: sparseBlocks }),
      });
      clearTimeout(timeout);
      if (!res.ok) return null;
      return (await res.json()) as NeuralChunkScanOutput;
    } catch {
      return null;
    }
  }

  /**
   * Queries the RTX 5060 Local SLM (Qwen2.5-Coder) for System 2 strategic decisions and tool synthesis.
   */
  public async generateLocalSLMResponse(
    systemPrompt: string,
    userPrompt: string,
    maxNewTokens = 320,
    repetitionPenalty = 1.1,
    temperature?: number,
    topP?: number,
    doSample?: boolean,
    botId = 'default'
  ): Promise<NeuralLLMOutput | null> {
    if (!(await this.checkHealth())) return null;

    try {
      const defaultTemp = process.env.SLM_TEMPERATURE ? parseFloat(process.env.SLM_TEMPERATURE) : 0.0;
      const defaultTopP = process.env.SLM_TOP_P ? parseFloat(process.env.SLM_TOP_P) : 1.0;
      const defaultSample = process.env.SLM_DO_SAMPLE === 'true' || defaultTemp > 0.0;

      const effectiveTemp = temperature !== undefined ? temperature : defaultTemp;
      const effectiveTopP = topP !== undefined ? topP : defaultTopP;
      const effectiveSample = doSample !== undefined ? doSample : (defaultSample || effectiveTemp > 0.0);

      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 35000);
      const res = await this.fetchFn(`${this.baseUrl}/llm/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          system_prompt: systemPrompt,
          user_prompt: userPrompt,
          max_new_tokens: maxNewTokens,
          repetition_penalty: repetitionPenalty,
          temperature: effectiveTemp,
          top_p: effectiveTopP,
          do_sample: effectiveSample,
          bot_id: botId,
        }),
      });
      clearTimeout(timeout);
      if (!res.ok) return null;
      const out = (await res.json()) as NeuralLLMOutput;
      console.log(
        `[RTX 5060 SLM] Generated ${out.tokens_generated} tokens in ${out.latency_ms}ms (${out.tokens_per_sec} tok/s)`
      );
      logVerbose(
        'LOCAL_GPU_SLM',
        `Generated ${out.tokens_generated} tokens in ${out.latency_ms}ms (${out.tokens_per_sec} tok/s)`,
        out.content
      );
      return out;
    } catch (err: any) {
      logVerbose('LOCAL_GPU_SLM_ERROR', `Local GPU SLM request failed or timed out: ${err?.message || err}`);
      return null;
    }
  }
}

export const sharedNeuralBridge = new NeuralBridgeClient();
